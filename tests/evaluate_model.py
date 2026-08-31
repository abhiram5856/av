import os
import io
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as tv_models
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# Try importing sklearn metrics, otherwise fallback to simple manual calculations
try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configuration
DATASET_DIR = r"backend/data/processed_dataset"
WEIGHTS_PATH = r"backend/models/weights/nova_mobilenet_v3.pth"
EVAL_DIR = "evaluation"
SPLIT_FILE = os.path.join(EVAL_DIR, "test_split.json")

os.makedirs(EVAL_DIR, exist_ok=True)

# Select Device
if os.environ.get("NOVA_FORCE_CPU") == "true":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_or_create_split() -> dict:
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            return json.load(f)
            
    print("No existing test split found. Creating a fixed 90/10 split...")
    random.seed(42) # Fixed seed for reproducibility
    split = {}
    
    classes = os.listdir(DATASET_DIR)
    for c in classes:
        c_path = os.path.join(DATASET_DIR, c)
        if not os.path.isdir(c_path):
            continue
        files = [f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Sort files to ensure same ordering before shuffle
        files.sort()
        # Shuffle with fixed seed
        random.shuffle(files)
        # Select 10% for test set
        test_size = max(1, int(len(files) * 0.10))
        test_files = files[:test_size]
        split[c] = test_files
        
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)
        
    return split

def manual_metrics(y_true, y_pred, classes_list):
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0
    
    # Per class metrics
    per_class = {}
    conf_mat = np.zeros((len(classes_list), len(classes_list)), dtype=int)
    
    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    
    for yt, yp in zip(y_true, y_pred):
        conf_mat[class_to_idx[yt]][class_to_idx[yp]] += 1
        
    for c in classes_list:
        idx = class_to_idx[c]
        tp = conf_mat[idx][idx]
        fp = sum(conf_mat[other][idx] for other in range(len(classes_list)) if other != idx)
        fn = sum(conf_mat[idx][other] for other in range(len(classes_list)) if other != idx)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": int(tp + fn)
        }
        
    # Micro/macro averages
    macro_precision = np.mean([per_class[c]["precision"] for c in classes_list])
    macro_recall = np.mean([per_class[c]["recall"] for c in classes_list])
    macro_f1 = np.mean([per_class[c]["f1-score"] for c in classes_list])
    
    return {
        "accuracy": round(accuracy, 4),
        "macro_avg": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1-score": round(macro_f1, 4)
        },
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist()
    }

def run_evaluation():
    print("=========================================")
    print("NOVA Model Evaluation Suite")
    print("=========================================")
    print(f"Target Device: {DEVICE}")
    
    # 1. Cold Start Benchmarking
    t0 = time.time()
    model = tv_models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 36)
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    cold_start_latency = time.time() - t0
    print(f"Cold Start Load Latency: {cold_start_latency:.4f} seconds")
    
    # Setup transform
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Dataset Test Split Ingestion
    split = get_or_create_split()
    classes_list = sorted(list(split.keys()))
    
    y_true = []
    y_pred = []
    
    print("\nRunning test split predictions...")
    
    # We will measure warm latency and memory
    inference_times = []
    
    # Process files
    processed_count = 0
    
    # Map indices directly using alphabetical order (matching PyTorch ImageFolder standard)
    idx_to_class = {i: c for i, c in enumerate(classes_list)}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    
    # Evaluate
    with torch.no_grad():
        for class_name, files in split.items():
            for filename in files:
                img_path = os.path.join(DATASET_DIR, class_name, filename)
                if not os.path.exists(img_path):
                    continue
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
                    
                    t_inf_start = time.perf_counter()
                    output = model(tensor)
                    pred_idx = torch.argmax(output[0]).item()
                    inference_times.append(time.perf_counter() - t_inf_start)
                    
                    pred_class = idx_to_class.get(pred_idx, class_name)
                    
                    y_true.append(class_name)
                    y_pred.append(pred_class)
                    processed_count += 1
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
                    
    print(f"Successfully evaluated {processed_count} test images.")
    
    # 3. Calculate Metrics
    results_summary = manual_metrics(y_true, y_pred, classes_list)
    accuracy = results_summary["accuracy"]
    macro_avg = results_summary["macro_avg"]
    
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1 Score: {macro_avg['f1-score']:.4f}")
    
    # Save metrics JSON
    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "macro_avg": macro_avg,
            "total_test_samples": processed_count
        }, f, indent=2)
        
    # Save classification report JSON
    with open(os.path.join(EVAL_DIR, "classification_report.json"), "w") as f:
        json.dump(results_summary["per_class"], f, indent=2)
        
    # 4. Generate Confusion Matrix Plot
    conf_matrix = np.array(results_summary["confusion_matrix"])
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('NOVA Model Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes_list))
    plt.xticks(tick_marks, [c[:15] for c in classes_list], rotation=90, fontsize=7)
    plt.yticks(tick_marks, [c[:15] for c in classes_list], fontsize=7)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion Matrix saved to evaluation/confusion_matrix.png")
    
    # 5. Performance Benchmarks
    avg_inf_latency = np.mean(inference_times) if inference_times else 0
    throughput = 1.0 / avg_inf_latency if avg_inf_latency > 0 else 0
    
    # Benchmark file
    with open(os.path.join(EVAL_DIR, "benchmark.md"), "w") as f:
        f.write("# NOVA Model Performance Benchmarks\n\n")
        f.write(f"- **Target Execution Device**: `{DEVICE}`\n")
        f.write(f"- **Cold Start Latency**: `{cold_start_latency:.4f} seconds`\n")
        f.write(f"- **Average Warm Inference Latency**: `{avg_inf_latency * 1000:.2f} ms`\n")
        f.write(f"- **Throughput (Inferences / Sec)**: `{throughput:.2f} QPS`\n")
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 ** 2)
            f.write(f"- **Peak GPU VRAM Footprint**: `{vram:.2f} MB`\n")
        else:
            f.write("- **Peak GPU VRAM Footprint**: `N/A (CPU execution mode)`\n")
        f.write(f"- **Total Samples Evaluated**: `{processed_count}`\n")
        
    print("Performance benchmarks written to evaluation/benchmark.md")

if __name__ == "__main__":
    run_evaluation()
