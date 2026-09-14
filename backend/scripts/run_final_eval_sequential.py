import os
import torch
import time
import json
from pathlib import Path
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import numpy as np

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
import sys
sys.path.append(str(REPO_ROOT))
from backend.models.class_registry import NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX, MODEL_CONFIG

def get_size(path):
    return os.path.getsize(path) / (1024 * 1024)

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
])

def evaluate_sequential(model_path, device, lab_samples, field_samples):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    def run_eval(samples):
        all_preds = []
        all_targets = []
        all_probs = []
        start = time.perf_counter()
        
        with torch.no_grad():
            for p, target in samples:
                img = Image.open(p).convert("RGB")
                img = val_transform(img).unsqueeze(0).to(device)
                out = model(img)
                _, pred = torch.max(out, 1)
                all_preds.append(pred.item())
                all_targets.append(target)
                all_probs.append(torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0])
                
        lat = (time.perf_counter() - start) / len(samples)
        return all_targets, all_preds, all_probs, lat

    lab_t, lab_p, lab_probs, _ = run_eval(lab_samples)
    field_t, field_p, field_probs, lat = run_eval(field_samples)
    
    return {
        "lab_acc": accuracy_score(lab_t, lab_p),
        "lab_f1": f1_score(lab_t, lab_p, average='macro', zero_division=0),
        "field_acc": accuracy_score(field_t, field_p),
        "field_f1": f1_score(field_t, field_p, average='macro', zero_division=0),
        "field_f1_weighted": f1_score(field_t, field_p, average='weighted', zero_division=0),
        "top3_acc": sum([1 if t in np.argsort(p)[-3:] else 0 for t, p in zip(field_t, field_probs)]) / len(field_t),
        "lat": lat,
        "size": get_size(model_path),
        "field_report": classification_report(field_t, field_p, target_names=CLASS_NAMES, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0),
        "field_cm": confusion_matrix(field_t, field_p, labels=list(range(NUM_CLASSES)))
    }

def print_top_confusions(cm, n=10):
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i][j] > 0:
                confusions.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    for true_c, pred_c, count in confusions[:n]:
        print(f"True: {true_c:35s} -> Pred: {pred_c:35s} ({count})")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Lab samples
    with open(REPO_ROOT / "evaluation" / "clean_test_split.json") as f:
        lab_data = json.load(f)
    lab_samples = []
    for item in lab_data:
        cls_name = item["class_name"]
        if cls_name in CLASS_TO_IDX:
            lab_samples.append((item["path"], CLASS_TO_IDX[cls_name]))
            
    # Load Field samples
    with open(REPO_ROOT / "backend" / "data" / "field_splits.json") as f:
        field_paths = json.load(f)["test"]
    field_samples = []
    for p in field_paths:
        cls_name = os.path.basename(os.path.dirname(p))
        if not os.path.isabs(p): 
            p = os.path.join(str(REPO_ROOT), "backend", "data", "processed_field_dataset", p)
        if cls_name in CLASS_TO_IDX:
            field_samples.append((p, CLASS_TO_IDX[cls_name]))
            
    print(f"Total Lab Samples: {len(lab_samples)}")
    print(f"Total Field Samples: {len(field_samples)}")
    
    base_path = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"
    cand_path = REPO_ROOT / "backend" / "models" / "weights" / "candidate_v3.pth"
    
    print("\nEvaluating Baseline...")
    base_res = evaluate_sequential(base_path, device, lab_samples, field_samples)
    print("Evaluating Candidate...")
    cand_res = evaluate_sequential(cand_path, device, lab_samples, field_samples)
    
    print("\n================== FINAL COMPARISON ==================")
    print(f"{'Metric':<25} {'Production':<12} {'Candidate':<12} {'Delta':<12}")
    
    def print_metric(name, b_val, c_val, is_pct=True):
        if is_pct:
            b_str, c_str = f"{b_val*100:.2f}%", f"{c_val*100:.2f}%"
            delta = (c_val - b_val)*100
            delta_str = f"{delta:+.2f}%"
        else:
            b_str, c_str = f"{b_val:.4f}", f"{c_val:.4f}"
            delta = c_val - b_val
            delta_str = f"{delta:+.4f}"
        print(f"{name:<25} {b_str:<12} {c_str:<12} {delta_str:<12}")

    print_metric("Field Accuracy", base_res['field_acc'], cand_res['field_acc'])
    print_metric("Field Macro F1", base_res['field_f1'], cand_res['field_f1'])
    print_metric("Field Weighted F1", base_res['field_f1_weighted'], cand_res['field_f1_weighted'])
    print_metric("Field Top-3", base_res['top3_acc'], cand_res['top3_acc'])
    print_metric("Lab Accuracy", base_res['lab_acc'], cand_res['lab_acc'])
    print_metric("Lab Macro F1", base_res['lab_f1'], cand_res['lab_f1'])
    print_metric("Inference Latency", base_res['lat'], cand_res['lat'], is_pct=False)
    print_metric("Model Size (MB)", base_res['size'], cand_res['size'], is_pct=False)
    
    print("\n--- BASELINE CONFUSIONS ---")
    print_top_confusions(base_res['field_cm'], 10)
    
    print("\n--- CANDIDATE CONFUSIONS ---")
    print_top_confusions(cand_res['field_cm'], 10)
    
    print("\n--- PER-CLASS FIELD METRICS (CANDIDATE vs BASELINE F1) ---")
    for cls in CLASS_NAMES:
        b_f1 = base_res['field_report'][cls]['f1-score']
        c_f1 = cand_res['field_report'][cls]['f1-score']
        if b_f1 > 0 or c_f1 > 0:
            print(f"{cls:40s} Base: {b_f1*100:5.1f}% -> Cand: {c_f1*100:5.1f}%  Diff: {(c_f1-b_f1)*100:+5.1f}%")

if __name__ == '__main__':
    main()
