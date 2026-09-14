import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import time

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
FIELD_DATA_DIR = REPO_ROOT / "backend" / "data" / "processed_field_dataset"
FIELD_SPLITS_PATH = REPO_ROOT / "backend" / "data" / "field_splits.json"
EXP_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "exp_mobilenet_v3.pth"

import sys
sys.path.append(str(REPO_ROOT))
from backend.models.class_registry import NUM_CLASSES, MODEL_CONFIG, CLASS_TO_IDX, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

class ExperimentDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['class_idx']

mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
size = MODEL_CONFIG["input_size"][0]
eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

with open(EVAL_DIR / "clean_test_split.json", "r") as f:
    lab_test = json.load(f)

with open(FIELD_SPLITS_PATH, "r") as f:
    field_splits = json.load(f)

field_test_items = []
for rel_path in field_splits["test"]:
    class_name = rel_path.split("/")[0]
    if class_name in CLASS_TO_IDX:
        field_test_items.append({
            "path": str(FIELD_DATA_DIR / rel_path),
            "class_name": class_name,
            "class_idx": CLASS_TO_IDX[class_name]
        })

test_field_dataset = ExperimentDataset(field_test_items, transform=eval_transforms)
test_lab_dataset = ExperimentDataset(lab_test, transform=eval_transforms)

test_field_loader = DataLoader(test_field_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_lab_loader = DataLoader(test_lab_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(EXP_WEIGHTS, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def full_eval(loader):
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    mac_f1 = f1_score(all_labels, all_preds, average='macro')
    wei_f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, mac_f1, wei_f1, all_labels, all_preds

print("\n=== FINAL EVALUATION (EPOCH 5 BEST CHECKPOINT) ===")
exp_field_acc, exp_field_mac_f1, exp_field_wei_f1, f_labels, f_preds = full_eval(test_field_loader)
exp_lab_acc, exp_lab_mac_f1, _, _, _ = full_eval(test_lab_loader)

base_field_acc = 0.7440
base_field_mac_f1 = 0.4660
base_lab_acc = 0.8197
base_lab_mac_f1 = 0.8686

print(f"\nMetric\t\t\tBaseline\tExperiment\tDelta")
print(f"Field Accuracy\t\t{base_field_acc:.4f}\t\t{exp_field_acc:.4f}\t\t{exp_field_acc - base_field_acc:+.4f}")
print(f"Field Macro F1\t\t{base_field_mac_f1:.4f}\t\t{exp_field_mac_f1:.4f}\t\t{exp_field_mac_f1 - base_field_mac_f1:+.4f}")
print(f"Lab Accuracy\t\t{base_lab_acc:.4f}\t\t{exp_lab_acc:.4f}\t\t{exp_lab_acc - base_lab_acc:+.4f}")
print(f"Lab Macro F1\t\t{base_lab_mac_f1:.4f}\t\t{exp_lab_mac_f1:.4f}\t\t{exp_lab_mac_f1 - base_lab_mac_f1:+.4f}")

report = classification_report(f_labels, f_preds, target_names=CLASS_NAMES, output_dict=True, labels=range(NUM_CLASSES), zero_division=0)
print("\n=== PER-CLASS FIELD RECALL & F1 (Experiment) ===")
class_f1s = []
for name in CLASS_NAMES:
    if name in report:
        class_f1s.append((name, report[name]['f1-score'], report[name]['recall'], report[name]['support']))
class_f1s.sort(key=lambda x: x[1])

print(f"{'Class':<50} {'F1':<10} {'Recall':<10} {'Support'}")
for name, f1, rec, sup in class_f1s:
    print(f"{name:<50} {f1:<10.4f} {rec:<10.4f} {sup}")
