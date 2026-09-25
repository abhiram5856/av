import sys
import os
import json
import time
import random
import psutil
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from backend.models.class_registry import NUM_CLASSES, CLASS_TO_IDX, MODEL_CONFIG
from backend.api.diagnose import tta_transforms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_memory_usage(prefix=""):
    cpu_mem = psutil.virtual_memory()
    gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    gpu_res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0
    print(f"[{prefix}] CPU RAM: {cpu_mem.percent}% ({cpu_mem.used / (1024**3):.1f}GB / {cpu_mem.total / (1024**3):.1f}GB) | GPU Alloc: {gpu_mem:.2f}GB | GPU Res: {gpu_res:.2f}GB")
    sys.stdout.flush()

class LazyImageDataset(Dataset):
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

def get_tta_eval(model, loader, device):
    all_targets, all_preds = [], []
    model.eval()
    with torch.inference_mode():
        for path, target in loader.dataset.samples:
            all_targets.append(target)
            with Image.open(path) as img:
                img = img.convert('RGB')
                tta_outs = []
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0).to(device, non_blocking=True)
                    tta_outs.append(torch.nn.functional.softmax(model(tensor), dim=1))
                
            avg = torch.stack(tta_outs).mean(dim=0)
            all_preds.append(torch.argmax(avg, dim=1).item())
            
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_targets, all_preds)
    mac = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, mac

def train():
    if not torch.cuda.is_available():
        print("CUDA is NOT available. Halting as per instructions.")
        return
        
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    FIELD_SPLITS_PATH = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    LAB_TRAIN_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_train_split.json")
    LAB_TEST_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_test_split.json")
    
    with open(FIELD_SPLITS_PATH) as f: field_splits = json.load(f)
    trainval_list = field_splits['trainval']
    field_test_list = field_splits['test']

    # Deterministic split 80/20 for field train/val
    trainval_list = sorted(list(set(trainval_list)))
    random.shuffle(trainval_list)
    split_idx = int(0.8 * len(trainval_list))
    field_train_list = trainval_list[:split_idx]
    field_val_list = trainval_list[split_idx:]

    lab_train_list = []
    with open(LAB_TRAIN_PATH) as f:
        lab_data = json.load(f)
        if isinstance(lab_data, dict):
            for v in lab_data.values(): lab_train_list.extend(v)
        else: lab_train_list = [i['path'] for i in lab_data]

    lab_test_list = []
    with open(LAB_TEST_PATH) as f:
        lab_test_data = json.load(f)
        if isinstance(lab_test_data, dict):
            for v in lab_test_data.values(): lab_test_list.extend(v)
        else: lab_test_list = [i['path'] for i in lab_test_data]

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
    ])
    
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
    ])

    field_train_ds = LazyImageDataset(field_train_list, train_tf)
    lab_train_ds = LazyImageDataset(lab_train_list, train_tf)
    
    # Controlled Sampling - We assign weights so Field samples are equally represented
    # but don't overwhelm. Lab: 1.0 weight, Field: (len(lab) / len(field)) weight * 0.5
    # Actually, simpler: Just concatenate, but use a proper dataset. Let's just concat for now.
    from torch.utils.data import ConcatDataset
    joint_ds = ConcatDataset([lab_train_ds, field_train_ds])
    
    field_val_ds = LazyImageDataset(field_val_list, eval_tf)
    field_test_ds = LazyImageDataset(field_test_list, eval_tf)
    lab_test_ds = LazyImageDataset(lab_test_list, eval_tf)

    batch_size = 8
    gradient_accumulation = 4
    
    train_loader = DataLoader(joint_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    field_val_loader = DataLoader(field_val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    lab_val_loader = DataLoader(lab_train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0) # Just for quick eval, we'll use a subset if needed

    print(f"\n--- SETUP ---")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Gradient Accumulation: {gradient_accumulation}")
    print(f"Effective Batch: {batch_size * gradient_accumulation}")
    print(f"Train Dataset: {len(joint_ds)} (Lab: {len(lab_train_ds)}, Field: {len(field_train_ds)})")
    print(f"Val Dataset: Field: {len(field_val_ds)}")
    print(f"Test Dataset: Field: {len(field_test_ds)}, Lab: {len(lab_test_ds)}")
    print_memory_usage("INIT")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    production_path = os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    model.load_state_dict(torch.load(production_path, map_location=device, weights_only=True))
    model.to(device)

    for param in model.features[:-3].parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # SMOKE TEST
    print("\n--- RUNNING SMOKE TEST ---")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(train_loader):
        if i >= 2: break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    print("Smoke test passed successfully!")
    print_memory_usage("SMOKE")

    best_val_acc = 0
    
    candidate_path = os.path.join(REPO_ROOT, "backend", "models", "weights", "candidate_field_domain_finetuned.pth")

    for epoch in range(10):
        print(f"\n--- EPOCH {epoch+1}/10 ---")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        train_losses = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(x)
                loss = criterion(out, y)
                loss = loss / gradient_accumulation
                
            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
            train_losses.append(loss.item() * gradient_accumulation)
            
            if i % 100 == 0:
                print_memory_usage(f"E{epoch+1} B{i}")
                
            # Memory safety GC
            del x, y, out, loss
            
        gc.collect()
        torch.cuda.empty_cache()

        model.eval()
        from sklearn.metrics import accuracy_score, f1_score
        
        # Field Val
        val_preds, val_targs = [], []
        with torch.inference_mode():
            for x, y in field_val_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(x)
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                val_targs.extend(y.numpy())
        f_v_acc = accuracy_score(val_targs, val_preds)
        f_v_mac = f1_score(val_targs, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Field Val Acc: {f_v_acc:.4f} | Field Val MacF1: {f_v_mac:.4f}")
        print_memory_usage(f"E{epoch+1} END")
        
        # Save epoch checkpoint
        torch.save(model.state_dict(), os.path.join(REPO_ROOT, f"candidate_epoch_{epoch+1:02d}.pth"))
        
        if f_v_acc > best_val_acc:
            best_val_acc = f_v_acc
            torch.save(model.state_dict(), candidate_path)
            print("-> Saved new best candidate!")

    print("\n--- FINAL TTA EVALUATION ---")
    
    prod_model = models.mobilenet_v3_small(weights=None)
    prod_model.classifier[3] = nn.Linear(prod_model.classifier[3].in_features, NUM_CLASSES)
    prod_model.load_state_dict(torch.load(production_path, map_location=device, weights_only=True))
    prod_model.to(device)
    
    p_f_acc, p_f_mac = get_tta_eval(prod_model, DataLoader(field_test_ds), device)
    p_l_acc, p_l_mac = get_tta_eval(prod_model, DataLoader(lab_test_ds), device)
    
    print(f"PRODUCTION - Field TTA Acc: {p_f_acc:.4f}, Macro: {p_f_mac:.4f}")
    print(f"PRODUCTION - Lab TTA Acc: {p_l_acc:.4f}, Macro: {p_l_mac:.4f}")

    if os.path.exists(candidate_path):
        cand_model = models.mobilenet_v3_small(weights=None)
        cand_model.classifier[3] = nn.Linear(cand_model.classifier[3].in_features, NUM_CLASSES)
        cand_model.load_state_dict(torch.load(candidate_path, map_location=device, weights_only=True))
        cand_model.to(device)
        
        c_f_acc, c_f_mac = get_tta_eval(cand_model, DataLoader(field_test_ds), device)
        c_l_acc, c_l_mac = get_tta_eval(cand_model, DataLoader(lab_test_ds), device)
        
        print(f"CANDIDATE - Field TTA Acc: {c_f_acc:.4f}, Macro: {c_f_mac:.4f}")
        print(f"CANDIDATE - Lab TTA Acc: {c_l_acc:.4f}, Macro: {c_l_mac:.4f}")
        
        if c_f_mac >= 0.707 and c_f_acc >= 0.70 and c_l_acc >= 0.85 and c_l_mac >= 0.85 and (c_f_mac > p_f_mac or c_f_acc > p_f_acc):
            print("DECISION: SAFE TO PROMOTE CANDIDATE")
        else:
            print("DECISION: KEEP CURRENT PRODUCTION")
            
        report_md = f"""# Field-Domain Fine-Tuning Final Report

## Hardware / Settings
- Device: CUDA
- GPU: {torch.cuda.get_device_name(0)}
- Batch Size: {batch_size} (Physical) x {gradient_accumulation} (Accumulation) = {batch_size * gradient_accumulation} (Effective)
- AMP Enabled: True
- Dataset Memory Strategy: Lazy Load (`Image.open` in `__getitem__` inside a context manager)

## Dataset Distribution
- Field Train: {len(field_train_ds)}
- Field Val: {len(field_val_ds)}
- Field Test: {len(field_test_ds)}
- Lab Train: {len(lab_train_ds)}
- Lab Val/Test: {len(lab_test_ds)}

## Production Baseline (5-View TTA)
- Field Accuracy: {p_f_acc:.4f}
- Field Macro F1: {p_f_mac:.4f}
- Lab Accuracy: {p_l_acc:.4f}
- Lab Macro F1: {p_l_mac:.4f}

## Candidate (5-View TTA)
- Field Accuracy: {c_f_acc:.4f}
- Field Macro F1: {c_f_mac:.4f}
- Lab Accuracy: {c_l_acc:.4f}
- Lab Macro F1: {c_l_mac:.4f}

## Decision
"""
        if c_f_mac >= 0.707 and c_f_acc >= 0.70 and c_l_acc >= 0.85 and c_l_mac >= 0.85 and (c_f_mac > p_f_mac or c_f_acc > p_f_acc):
            report_md += "**SAFE TO PROMOTE CANDIDATE**"
        else:
            report_md += "**KEEP CURRENT PRODUCTION**"
            
        with open(os.path.join(REPO_ROOT, "evaluation", "field_domain_finetuning_final.md"), "w") as f:
            f.write(report_md)
            
    else:
        print("Candidate not found.")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        with open('crash_log.txt', 'w') as f:
            f.write(traceback.format_exc())
        raise
