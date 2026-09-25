import sys
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from PIL import Image
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

class SimpleImageDataset(Dataset):
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
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

def get_tta_eval(model, loader, device):
    all_targets, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for path, target in loader.dataset.samples:
            all_targets.append(target)
            img = Image.open(path).convert('RGB')
            tta_outs = []
            for t in tta_transforms:
                tensor = t(img).unsqueeze(0).to(device)
                tta_outs.append(torch.nn.functional.softmax(model(tensor), dim=1))
            avg = torch.stack(tta_outs).mean(dim=0)
            all_preds.append(torch.argmax(avg, dim=1).item())
            
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_targets, all_preds)
    mac = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, mac

def train():
    device = torch.device('cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    FIELD_SPLITS_PATH = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    LAB_TRAIN_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_train_split.json")
    LAB_TEST_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_test_split.json")
    
    with open(FIELD_SPLITS_PATH) as f: field_splits = json.load(f)
    trainval_list = field_splits['trainval']
    field_test_list = field_splits['test']

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

    field_train_ds = SimpleImageDataset(field_train_list, train_tf)
    lab_train_ds = SimpleImageDataset(lab_train_list, train_tf)
    joint_ds = ConcatDataset([lab_train_ds, field_train_ds])
    
    field_val_ds = SimpleImageDataset(field_val_list, eval_tf)
    field_test_ds = SimpleImageDataset(field_test_list, eval_tf)
    lab_test_ds = SimpleImageDataset(lab_test_list, eval_tf)

    batch_size = 16
    train_loader = DataLoader(joint_ds, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(field_val_ds, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    production_path = os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    model.load_state_dict(torch.load(production_path, map_location=device, weights_only=True))
    model.to(device)

    for param in model.features[:-3].parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    candidate_path = os.path.join(REPO_ROOT, "backend", "models", "weights", "candidate_field_domain_finetuned.pth")

    for epoch in range(10):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_targs = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                val_targs.extend(y.cpu().numpy())
                
        from sklearn.metrics import accuracy_score
        v_acc = accuracy_score(val_targs, val_preds)
        print(f"Epoch {epoch} | Val Acc: {v_acc:.4f}")
        sys.stdout.flush()
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), candidate_path)

    print("\n--- FINAL TTA EVALUATION ---")
    sys.stdout.flush()
    
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
    else:
        print("Candidate not found.")

if __name__ == '__main__':
    try:
        train()
    except Exception as e:
        with open('crash_log.txt', 'w') as f:
            f.write(traceback.format_exc())
        raise
