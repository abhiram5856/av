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

BASE_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"
EXP_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "candidate_v3.pth"

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
        if self.transform: img = self.transform(img)
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

print(f"Verified Active Field Test Size: {len(field_test_items)}")

test_field_dataset = ExperimentDataset(field_test_items, transform=eval_transforms)
test_lab_dataset = ExperimentDataset(lab_test, transform=eval_transforms)
test_field_loader = DataLoader(test_field_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_lab_loader = DataLoader(test_lab_dataset, batch_size=BATCH_SIZE, shuffle=False)

def build_model(weights_path):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def evaluate(model, loader):
    all_preds, all_labels, all_probs = [], [], []
    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    latency_ms = ((time.time() - start_time) / max(1, len(all_labels))) * 1000
    acc = accuracy_score(all_labels, all_preds)
    mac_f1 = f1_score(all_labels, all_preds, average='macro')
    wei_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    top3 = sum(1 for i, p in enumerate(all_probs) if all_labels[i] in np.argsort(p)[-3:])
    top3_acc = top3 / max(1, len(all_labels))
    
    return acc, mac_f1, wei_f1, top3_acc, latency_ms, all_labels, all_preds

# 1. BASELINE
model_base = build_model(BASE_WEIGHTS)
b_f_acc, b_f_mac_f1, b_f_wei_f1, b_f_top3, b_lat, _, _ = evaluate(model_base, test_field_loader)
b_l_acc, b_l_mac_f1, b_l_wei_f1, b_l_top3, _, _, _ = evaluate(model_base, test_lab_loader)
del model_base

# 2. EXPERIMENT
model_exp = build_model(EXP_WEIGHTS)
e_f_acc, e_f_mac_f1, e_f_wei_f1, e_f_top3, e_lat, f_labels, f_preds = evaluate(model_exp, test_field_loader)
e_l_acc, e_l_mac_f1, e_l_wei_f1, e_l_top3, _, _, _ = evaluate(model_exp, test_lab_loader)

print(f"\n--- REPRODUCED METRICS ---")
print(f"Field Accuracy: BASE={b_f_acc:.4f} | EXP={e_f_acc:.4f}")
print(f"Field Macro F1: BASE={b_f_mac_f1:.4f} | EXP={e_f_mac_f1:.4f}")
print(f"Field Weighted F1: BASE={b_f_wei_f1:.4f} | EXP={e_f_wei_f1:.4f}")
print(f"Field Top-3: BASE={b_f_top3:.4f} | EXP={e_f_top3:.4f}")
print(f"Lab Accuracy: BASE={b_l_acc:.4f} | EXP={e_l_acc:.4f}")
print(f"Lab Macro F1: BASE={b_l_mac_f1:.4f} | EXP={e_l_mac_f1:.4f}")
print(f"Latency: BASE={b_lat:.2f}ms | EXP={e_lat:.2f}ms")

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

cm = confusion_matrix(f_labels, f_preds, labels=range(NUM_CLASSES))
print("\n=== TOP CONFUSIONS ===")
confs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i,j] > 0:
            confs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i,j]))
confs.sort(key=lambda x: x[2], reverse=True)
for true_c, pred_c, count in confs[:5]:
    print(f"True: {true_c} -> Pred: {pred_c} ({count})")
