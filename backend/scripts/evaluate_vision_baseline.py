import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI"))

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG, idx_to_class

# Configuration
EVAL_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\evaluation")
TEST_SPLIT_PATH = os.path.join(EVAL_DIR, "test_split.json")
WEIGHTS_PATH = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights\best_baseline.pth")

class JsonDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.records = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = record["path"]
        label = record["class_idx"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupted
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def build_model(num_classes=NUM_CLASSES):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def evaluate_model():
    if not os.path.exists(TEST_SPLIT_PATH):
        print(f"Test split not found at {TEST_SPLIT_PATH}")
        return
        
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Weights not found at {WEIGHTS_PATH}. Did you run training?")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Same val/test transform as in training
    mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
    size = MODEL_CONFIG["input_size"][0]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dataset = JsonDataset(TEST_SPLIT_PATH, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_paths = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            top_probs, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    
    # Top-3 Accuracy
    top3_correct = 0
    for i, target in enumerate(all_targets):
        top3_idx = np.argsort(all_probs[i])[-3:]
        if target in top3_idx:
            top3_correct += 1
    top3_acc = top3_correct / len(all_targets)

    # Precision, Recall, F1 (Macro & Weighted)
    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    precision_wt, recall_wt, f1_wt, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    
    # Per-class metrics
    precision_cls, recall_cls, f1_cls, support = precision_recall_fscore_support(all_targets, all_preds, labels=range(NUM_CLASSES), zero_division=0)
    
    per_class_metrics = []
    for i in range(NUM_CLASSES):
        per_class_metrics.append({
            "class": CLASS_NAMES[i],
            "precision": float(precision_cls[i]),
            "recall": float(recall_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(support[i])
        })

    # Confidence Analysis
    correct_mask = (all_preds == all_targets)
    incorrect_mask = ~correct_mask
    
    confidences = np.max(all_probs, axis=1)
    
    # High confidence incorrect predictions
    high_conf_incorrect = []
    for i in np.where(incorrect_mask)[0]:
        if confidences[i] > 0.8:  # Arbitrary high confidence threshold
            high_conf_incorrect.append({
                "path": all_paths[i],
                "true_class": CLASS_NAMES[all_targets[i]],
                "predicted_class": CLASS_NAMES[all_preds[i]],
                "confidence": float(confidences[i])
            })

    metrics = {
        "accuracy": float(acc),
        "top3_accuracy": float(top3_acc),
        "macro_precision": float(precision_mac),
        "macro_recall": float(recall_mac),
        "macro_f1": float(f1_mac),
        "weighted_f1": float(f1_wt),
        "confidence_analysis": {
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "mean_confidence_correct": float(np.mean(confidences[correct_mask])) if sum(correct_mask) > 0 else 0,
            "mean_confidence_incorrect": float(np.mean(confidences[incorrect_mask])) if sum(incorrect_mask) > 0 else 0,
            "num_high_confidence_errors_gt_0.8": len(high_conf_incorrect)
        },
        "per_class": per_class_metrics
    }

    # Save metrics
    metrics_path = os.path.join(EVAL_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save high conf incorrect
    with open(os.path.join(EVAL_DIR, "high_conf_errors.json"), "w") as f:
        json.dump(high_conf_incorrect, f, indent=2)
        
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='g', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {acc:.2f}, F1: {f1_mac:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
    
    print(f"Evaluation complete. Metrics saved to {EVAL_DIR}")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1_mac:.4f} | Top-3 Acc: {top3_acc:.4f}")

if __name__ == "__main__":
    evaluate_model()
