import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import time
from collections import Counter
import copy

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
BASE_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"
EXP_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "candidate_v3.pth"

import sys
sys.path.append(str(REPO_ROOT))
from backend.models.class_registry import NUM_CLASSES, MODEL_CONFIG, CLASS_TO_IDX, CLASS_NAMES

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
HARD_MULTIPLIER = 1.5

HARD_CLASSES = {
    "rice_brown_spot",
    "rice_leaf_blast",
    "rice_bacterial_leaf_blight",
    "rice_healthy"
}

# ==========================================
# DATASET DEFINITION
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            cls_name = item["class_name"]
            p = item["path"]
            if cls_name in CLASS_TO_IDX:
                self.samples.append((p, CLASS_TO_IDX[cls_name]))
            
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# AUGMENTATIONS
# ==========================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
])

# ==========================================
# MIXUP UTILS
# ==========================================
def mixup_data(x, y, alpha=0.2, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# TRAINING SCRIPT
# ==========================================
def main():
    print("Loading data...")
    train_dataset = SimpleDataset(EVAL_DIR / "clean_train_split.json", transform=train_transform)
    val_dataset = SimpleDataset(EVAL_DIR / "clean_val_split.json", transform=val_transform)
    
    # Calculate weights
    labels = [s[1] for s in train_dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)
    
    weights = []
    print("\n--- Effective Sampling Weights ---")
    for cls_idx in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_idx]
        count = class_counts.get(cls_idx, 0)
        if count == 0:
            w = 0.0
        else:
            w = total / (NUM_CLASSES * count)
            if cls_name in HARD_CLASSES:
                w *= HARD_MULTIPLIER
        weights.append(w)
        print(f"{cls_name:40s} Weight: {w:.4f} (Count: {count})")
        
    sample_weights = [weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load Model
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    
    print(f"Loading base weights from {BASE_WEIGHTS}...")
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=device, weights_only=True))
    
    # Freeze everything except last features and classifier
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.features[-2:].parameters():
        param.requires_grad = True
        
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_val_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # MixUp
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, MIXUP_ALPHA, device)
            
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        all_preds = []
        all_targs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targs.extend(targets.cpu().numpy())
                
        val_f1 = f1_score(all_targs, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), EXP_WEIGHTS)
            print(f" -> Saved new best candidate with Val F1: {val_f1:.4f}")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
    print(f"Candidate saved to: {EXP_WEIGHTS}")

if __name__ == '__main__':
    main()
