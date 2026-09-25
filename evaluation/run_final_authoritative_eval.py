import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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
    start_time = time.time()
    
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
            
    latency_ms = ((time.time() - start_time) / len(loader.dataset)) * 1000
    
    acc = accuracy_score(all_targets, all_preds)
    mac = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    wei = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    cr = classification_report(all_targets, all_preds, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0)
    
    return acc, mac, wei, cm, cr, latency_ms

def eval_no_tta(model, loader, device):
    all_targets, all_preds = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_targets.extend(y.numpy())
    cr = classification_report(all_targets, all_preds, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0)
    return cr

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
    
    # We need to evaluate the validation set to investigate the F1 drop
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
    
    # 1. Investigate Epoch 9 vs Epoch 10 on Validation Set (where the drop happened)
    print("Investigating Epoch 9 vs Epoch 10 Drop on Validation Set...")
    ep9_model = load_model(os.path.join(REPO_ROOT, "candidate_epoch_09.pth"), device)
    ep10_model = load_model(os.path.join(REPO_ROOT, "candidate_epoch_10.pth"), device)
    
    cr9 = eval_no_tta(ep9_model, field_val_loader, device)
    cr10 = eval_no_tta(ep10_model, field_val_loader, device)
    
    drop_report = "### Epoch 9 vs Epoch 10 Investigation (Validation Set)\n"
    for cls_idx in range(NUM_CLASSES):
        str_idx = str(cls_idx)
        if str_idx in cr9 and str_idx in cr10:
            if cr9[str_idx]['support'] > 0:
                f1_9 = cr9[str_idx]['f1-score']
                f1_10 = cr10[str_idx]['f1-score']
                rec_9 = cr9[str_idx]['recall']
                rec_10 = cr10[str_idx]['recall']
                if abs(f1_9 - f1_10) > 0.1:
                    drop_report += f"- **{IDX_TO_CLASS[cls_idx]}** (Support: {cr9[str_idx]['support']}): F1 {f1_9:.4f} -> {f1_10:.4f}, Recall {rec_9:.4f} -> {rec_10:.4f}\n"

    # 2. Authoritative TTA Evaluation
    print("Running authoritative TTA evaluation...")
    prod_model = load_model(os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth"), device)
    # Selected candidate is Epoch 10 based on validation accuracy (88.67% vs 88.56%)
    cand_model = ep10_model 
    
    p_f_acc, p_f_mac, p_f_wei, p_f_cm, p_f_cr, p_f_lat = eval_tta(prod_model, field_test_loader, device)
    p_l_acc, p_l_mac, p_l_wei, _, _, p_l_lat = eval_tta(prod_model, lab_test_loader, device)
    
    c_f_acc, c_f_mac, c_f_wei, c_f_cm, c_f_cr, c_f_lat = eval_tta(cand_model, field_test_loader, device)
    c_l_acc, c_l_mac, c_l_wei, _, _, c_l_lat = eval_tta(cand_model, lab_test_loader, device)
    
    print("Generating markdown report...")
    md = f"""# Final Authoritative Evaluation Report

## 1. Production Baseline (nova_mobilenet_v3_34_classes.pth)
- **Field Accuracy**: {p_f_acc:.4f}
- **Field Macro F1**: {p_f_mac:.4f}
- **Field Weighted F1**: {p_f_wei:.4f}
- **Field Latency (TTA)**: {p_f_lat:.2f} ms/image

- **Lab Accuracy**: {p_l_acc:.4f}
- **Lab Macro F1**: {p_l_mac:.4f}
- **Lab Weighted F1**: {p_l_wei:.4f}
- **Lab Latency (TTA)**: {p_l_lat:.2f} ms/image

## 2. Selected Candidate Checkpoint
Selected checkpoint: `candidate_epoch_10.pth` (Highest validation accuracy: 88.67%).

## 3. Field Test Metrics (Candidate)
- **Field Accuracy**: {c_f_acc:.4f}
- **Field Macro F1**: {c_f_mac:.4f}
- **Field Weighted F1**: {c_f_wei:.4f}
- **Field Latency (TTA)**: {c_f_lat:.2f} ms/image

## 4. Lab Test Metrics (Candidate)
- **Lab Accuracy**: {c_l_acc:.4f}
- **Lab Macro F1**: {c_l_mac:.4f}
- **Lab Weighted F1**: {c_l_wei:.4f}
- **Lab Latency (TTA)**: {c_l_lat:.2f} ms/image

## 5. Investigation of Epoch 9 -> Epoch 10 Macro F1 Drop
{drop_report}

## 6. Confusion Matrix (Candidate Field Test)
```
{np.array2string(c_f_cm, max_line_width=200)}
```
"""
    with open(os.path.join(REPO_ROOT, "evaluation", "final_authoritative_report.md"), "w") as f:
        f.write(md)
        
    print("Done!")

if __name__ == '__main__':
    main()
