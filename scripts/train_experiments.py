import os
import copy
import json
import argparse
import logging
import time
from pathlib import Path
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
EXP_DIR = REPO_ROOT / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

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
        label = item['class_idx']
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(exp_type):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    if exp_type == 'C': # Field Robust
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            # Add simple random shadow / blur approximation
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))], p=0.3),
            transforms.Normalize(mean, std),
        ])
    else: # Standard A, B, D
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
    return train_transform, val_transform

def build_model(model_name, num_classes):
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {model_name}")
    return model

def freeze_backbone(model, model_name):
    if 'mobilenet' in model_name:
        for param in model.features.parameters():
            param.requires_grad = False
    elif 'efficientnet' in model_name:
        for param in model.features.parameters():
            param.requires_grad = False

def unfreeze_backbone(model, model_name):
    if 'mobilenet' in model_name:
        for param in model.features.parameters():
            param.requires_grad = True
    elif 'efficientnet' in model_name:
        for param in model.features.parameters():
            param.requires_grad = True

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, correct / total

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / total, correct / total, all_preds, all_labels

def calculate_macro_f1(preds, labels, num_classes):
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average='macro', zero_division=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, choices=['A', 'B', 'C', 'D_large', 'D_effnet'])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Starting Experiment {args.exp} on {device}")
    
    # 1. Datasets & Loaders
    train_transform, val_transform = get_transforms(args.exp)
    
    train_dataset = CleanSplitDataset(EVAL_DIR / 'clean_train_split.json', transform=train_transform)
    val_dataset = CleanSplitDataset(EVAL_DIR / 'clean_val_split.json', transform=val_transform)
    
    with open(EVAL_DIR / 'dataset_audit_report.json', 'r') as f:
        audit_report = json.load(f)
    num_classes = audit_report['total_classes']
    
    # Imbalance weighting
    train_labels = [item['class_idx'] for item in train_dataset.data]
    counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 2. Model setup
    model_name = 'mobilenet_v3_small'
    if args.exp == 'D_large': model_name = 'mobilenet_v3_large'
    elif args.exp == 'D_effnet': model_name = 'efficientnet_b0'
    
    model = build_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history = []
    best_f1 = 0.0
    best_weights = None
    
    # 3. Training Strategy
    if args.exp in ['A', 'D_large', 'D_effnet']:
        # Head only or standard fine-tuning (we'll just freeze backbone for A, unfreeze for D)
        if args.exp == 'A':
            freeze_backbone(model, model_name)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(args.epochs):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, preds, labels = val_epoch(model, val_loader, criterion, device)
            f1 = calculate_macro_f1(preds, labels, num_classes)
            
            logger.info(f"Epoch {epoch+1:02d} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f} | Val F1: {f1:.4f}")
            history.append({'epoch': epoch, 'val_acc': v_acc, 'val_f1': f1})
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = copy.deepcopy(model.state_dict())
                
    elif args.exp in ['B', 'C']:
        # Progressive Fine-tuning
        # Stage 1: Head only
        freeze_backbone(model, model_name)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(5):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, preds, labels = val_epoch(model, val_loader, criterion, device)
            f1 = calculate_macro_f1(preds, labels, num_classes)
            logger.info(f"Stage 1 - Epoch {epoch+1:02d} | Val Acc: {v_acc:.4f} | Val F1: {f1:.4f}")
            history.append({'stage': 1, 'epoch': epoch, 'val_acc': v_acc, 'val_f1': f1})
            if f1 > best_f1:
                best_f1 = f1
                best_weights = copy.deepcopy(model.state_dict())
                
        # Stage 2: Unfreeze all, lower LR
        unfreeze_backbone(model, model_name)
        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': 1e-5},
            {'params': model.classifier.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-4)
        
        for epoch in range(args.epochs - 5):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            v_loss, v_acc, preds, labels = val_epoch(model, val_loader, criterion, device)
            f1 = calculate_macro_f1(preds, labels, num_classes)
            logger.info(f"Stage 2 - Epoch {epoch+1:02d} | Val Acc: {v_acc:.4f} | Val F1: {f1:.4f}")
            history.append({'stage': 2, 'epoch': epoch, 'val_acc': v_acc, 'val_f1': f1})
            if f1 > best_f1:
                best_f1 = f1
                best_weights = copy.deepcopy(model.state_dict())

    # 4. Save results
    exp_out = EXP_DIR / f"exp_{args.exp}"
    exp_out.mkdir(exist_ok=True)
    
    if best_weights:
        torch.save(best_weights, exp_out / "best_model.pth")
        
    with open(exp_out / "history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    logger.info(f"Experiment {args.exp} completed. Best Val F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
