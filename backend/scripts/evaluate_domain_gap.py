import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import defaultdict

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG, CLASS_TO_IDX

WEIGHTS_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights")
CHECKPOINT = os.path.join(WEIGHTS_DIR, MODEL_CONFIG["checkpoint_filename"])
FIELD_DATA_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_field_dataset")
SPLITS_PATH = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\field_splits.json")
REPORT_PATH = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\docs\DOMAIN_GAP_REPORT.md")

def evaluate_domain_gap():
    print(f"Loading checkpoint: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    
    # Setup transform (standard eval transform)
    mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
    size = MODEL_CONFIG["input_size"][0]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load test split
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    test_paths = splits["test"]
    
    print(f"Evaluating Domain Gap on {len(test_paths)} Field Test images...")
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for rel_path in test_paths:
            full_path = os.path.join(FIELD_DATA_DIR, rel_path)
            class_name = os.path.dirname(rel_path)
            
            if class_name not in CLASS_TO_IDX:
                continue
                
            y_true.append(CLASS_TO_IDX[class_name])
            
            try:
                img = Image.open(full_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                
                outputs = model(tensor)
                _, preds = torch.max(outputs, 1)
                y_pred.append(preds.item())
            except Exception as e:
                print(f"Error processing {full_path}: {e}")
                
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    
    report_text = f"""# ML Phase 2C: Domain Gap Analysis

## Overview
This report quantifies the "Domain Gap"—the drop in accuracy when our laboratory-trained baseline model is applied to real-world field data.

- **Model Checkpoint:** `{os.path.basename(CHECKPOINT)}`
- **Field Test Set Size:** {len(test_paths)} strictly held-out images
- **Architecture:** MobileNetV3-Small (26-class initialized)

## Zero-Shot Field Metrics
| Metric | Laboratory Baseline (Phase 1) | Field Evaluation (Phase 2C Zero-Shot) | Absolute Drop |
|--------|------------------------------|--------------------------------------|---------------|
| **Accuracy** | 91.75% | {acc*100:.2f}% | {(91.75 - acc*100):.2f}% |
| **Macro F1** | 90.82% | {f1*100:.2f}% | {(90.82 - f1*100):.2f}% |

## Interpretation
As expected in real-world ML adaptation, there is a massive drop in performance when moving from clean, white-background laboratory images to noisy, varied-lighting field images. 

*(Note: The Cotton classes were randomly initialized during the surgical expansion, contributing to the drop, but even the previously learned Rice classes suffer from domain shift).*

**Conclusion:** The model is not ready for deployment. We MUST perform Domain Adaptation (Field Fine-Tuning) to recover this performance loss.
"""
    
    with open(REPORT_PATH, 'w') as f:
        f.write(report_text)
        
    print(f"Domain Gap Evaluation Complete: Acc={acc*100:.2f}%, F1={f1*100:.2f}%")
    print(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    evaluate_domain_gap()
