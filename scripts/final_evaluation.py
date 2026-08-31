import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import time

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
EXP_DIR = REPO_ROOT / "experiments"
WEIGHTS_PATH = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"

class CleanSplitDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['path']).convert('RGB')
        label = int(item['class_idx'])
        path = item['path']
        if self.transform:
            img = self.transform(img)
        return img, label, path

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def build_model(num_classes, weights_path, device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(EVAL_DIR / 'dataset_audit_report.json', 'r') as f:
        audit_report = json.load(f)
    num_classes = audit_report['total_classes']
    idx_to_class = {int(v): k for k, v in audit_report['class_mapping'].items()}
    
    test_dataset = CleanSplitDataset(EVAL_DIR / 'clean_test_split.json', transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    if not WEIGHTS_PATH.exists():
        print(f"Weights not found at {WEIGHTS_PATH}")
        return
        
    model = build_model(num_classes, WEIGHTS_PATH, device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []
    
    print("Running inference on test set...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
            if i % 10 == 0:
                print(f"Batch {i}/{len(test_loader)}")
            
    total_time = time.time() - start_time
    latency_ms = (total_time / len(test_dataset)) * 1000
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate Top-3 Accuracy
    top3_correct = 0
    for i in range(len(all_labels)):
        top3_preds = np.argsort(all_probs[i])[-3:]
        if all_labels[i] in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(all_labels)
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    model_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
    
    print(f"\nFinal Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Inference Latency: {latency_ms:.2f} ms")
    print(f"Model Size: {model_size_mb:.2f} MB")
    
    metrics = {
        "Accuracy": acc,
        "Top-3 Accuracy": top3_acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Macro Precision": macro_p,
        "Macro Recall": macro_r,
        "Inference Latency ms": latency_ms,
        "Model Size MB": model_size_mb
    }
    
    with open(EVAL_DIR / "final_34_class_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Classification Report
    target_names = [idx_to_class[i] for i in range(num_classes)]
    clf_report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    with open(EVAL_DIR / "final_34_class_classification_report.json", "w") as f:
        json.dump(clf_report, f, indent=2)
        
    # High-confidence errors
    confidences = np.max(all_probs, axis=1)
    df = pd.DataFrame({
        "Image_Path": all_paths,
        "True_Label": [idx_to_class[i] for i in all_labels],
        "Predicted_Label": [idx_to_class[i] for i in all_preds],
        "Confidence": confidences,
        "Is_Correct": all_labels == all_preds
    })
    
    errors = df[df["Is_Correct"] == False]
    high_conf_errors = errors[errors["Confidence"] > 0.8]
    high_conf_errors = high_conf_errors.sort_values(by="Confidence", ascending=False)
    high_conf_errors.to_csv(EVAL_DIR / "final_34_class_high_confidence_errors.csv", index=False)
    print(f"\nFound {len(high_conf_errors)} high-confidence (>0.8) errors.")
    
    # Weakest classes
    class_f1s = {k: v['f1-score'] for k, v in clf_report.items() if k in target_names}
    weakest = sorted(class_f1s.items(), key=lambda x: x[1])[:5]
    print("\nFive Weakest Classes (F1 Score):")
    for cls, f1 in weakest:
        print(f" - {cls}: {f1:.4f}")

if __name__ == "__main__":
    main()
