import sys
import os
import json
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import hashlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from backend.models.class_registry import NUM_CLASSES, CLASS_TO_IDX, MODEL_CONFIG
from backend.api.diagnose import tta_transforms

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class EvalDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.samples = []
        for p in paths:
            cls_name = os.path.basename(os.path.dirname(p))
            if cls_name in CLASS_TO_IDX:
                if not os.path.isabs(p):
                    p_field = os.path.join(REPO_ROOT, "backend", "data", "processed_field_dataset", p)
                    p_lab = os.path.join(REPO_ROOT, p)
                    if os.path.exists(p_field): p = p_field
                    elif os.path.exists(p_lab): p = p_lab
                if os.path.exists(p):
                    self.samples.append((p, CLASS_TO_IDX[cls_name]))
        self.transform = transform
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label

def load_model(path, device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def eval_tta(model, loader, device):
    all_targets, all_preds = [], []
    with torch.inference_mode():
        for path, target in loader.dataset.samples:
            all_targets.append(target)
            with Image.open(path) as img:
                img = img.convert('RGB')
                tta_outs = []
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0).to(device)
                    tta_outs.append(torch.nn.functional.softmax(model(tensor), dim=1))
            avg = torch.stack(tta_outs).mean(dim=0)
            all_preds.append(torch.argmax(avg, dim=1).item())
            
    acc = accuracy_score(all_targets, all_preds)
    mac = f1_score(all_targets, all_preds, average='macro', labels=list(range(NUM_CLASSES)), zero_division=0)
    wei = f1_score(all_targets, all_preds, average='weighted', labels=list(range(NUM_CLASSES)), zero_division=0)
    cr = classification_report(all_targets, all_preds, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0)
    
    return acc, mac, wei, cr

def eval_no_tta(model, loader, device):
    all_targets, all_preds = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_targets.extend(y.numpy())
    acc = accuracy_score(all_targets, all_preds)
    mac = f1_score(all_targets, all_preds, average='macro', labels=list(range(NUM_CLASSES)), zero_division=0)
    mac_present = f1_score(all_targets, all_preds, average='macro', zero_division=0) # Averaged only over present classes
    return acc, mac, mac_present

def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
    ])
    
    FIELD_SPLITS_PATH = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    LAB_TEST_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_test_split.json")
    
    with open(FIELD_SPLITS_PATH) as f: field_splits = json.load(f)
    field_test_list = field_splits['test']
    
    # Val split for Ep9 vs Ep10
    trainval_list = sorted(list(set(field_splits['trainval'])))
    import random
    random.seed(42)
    random.shuffle(trainval_list)
    split_idx = int(0.8 * len(trainval_list))
    field_val_list = trainval_list[split_idx:]
    
    lab_test_list = []
    with open(LAB_TEST_PATH) as f:
        lab_test_data = json.load(f)
        if isinstance(lab_test_data, dict):
            for v in lab_test_data.values(): lab_test_list.extend(v)
        else: lab_test_list = [i['path'] for i in lab_test_data]

    field_test_ds = EvalDataset(field_test_list, eval_tf)
    lab_test_ds = EvalDataset(lab_test_list, eval_tf)
    field_val_ds = EvalDataset(field_val_list, eval_tf)
    
    field_test_loader = DataLoader(field_test_ds, batch_size=32)
    lab_test_loader = DataLoader(lab_test_ds, batch_size=32)
    field_val_loader = DataLoader(field_val_ds, batch_size=32)

    results = {}
    
    checkpoints = [os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")]
    for i in range(1, 11):
        p = os.path.join(REPO_ROOT, f"candidate_epoch_{i:02d}.pth")
        if os.path.exists(p): checkpoints.append(p)
    p_final = os.path.join(REPO_ROOT, "backend", "models", "weights", "candidate_field_domain_finetuned.pth")
    if os.path.exists(p_final): checkpoints.append(p_final)
    checkpoints = list(set(checkpoints))

    print("Evaluating checkpoints on test sets (TTA)...")
    for cp in checkpoints:
        name = os.path.basename(cp)
        print(f"Evaluating {name}...")
        model = load_model(cp, device)
        f_acc, f_mac, f_wei, _ = eval_tta(model, field_test_loader, device)
        l_acc, l_mac, l_wei, cr = eval_tta(model, lab_test_loader, device)
        results[name] = {
            'field_acc': f_acc, 'field_mac': f_mac, 'field_wei': f_wei,
            'lab_acc': l_acc, 'lab_mac': l_mac, 'lab_wei': l_wei,
            'lab_cr': cr,
            'sha256': sha256_checksum(cp)
        }

    print("Evaluating Ep9 vs Ep10 anomaly on Validation set (No TTA)...")
    val_anomaly = {}
    for name in ["candidate_epoch_09.pth", "candidate_epoch_10.pth"]:
        cp = os.path.join(REPO_ROOT, name)
        if os.path.exists(cp):
            model = load_model(cp, device)
            v_acc, v_mac, v_mac_present = eval_no_tta(model, field_val_loader, device)
            val_anomaly[name] = {
                'val_acc': v_acc, 'val_mac_34': v_mac, 'val_mac_present': v_mac_present
            }

    with open(os.path.join(REPO_ROOT, "evaluation", "audit_data.json"), "w") as f:
        json.dump({'checkpoints': results, 'val_anomaly': val_anomaly}, f, indent=4)
        
if __name__ == '__main__':
    main()
