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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import time
from collections import Counter
import random

# ==========================================
# CONFIGURATION
# ==========================================
REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
FIELD_DATA_DIR = REPO_ROOT / "backend" / "data" / "processed_field_dataset"
FIELD_SPLITS_PATH = REPO_ROOT / "backend" / "data" / "field_splits.json"
BASE_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"
EXP_WEIGHTS = REPO_ROOT / "backend" / "models" / "weights" / "exp_mobilenet_v3.pth"

import sys
sys.path.append(str(REPO_ROOT))
from backend.models.class_registry import NUM_CLASSES, MODEL_CONFIG, CLASS_TO_IDX, CLASS_NAMES

EPOCHS = 10
BATCH_SIZE = 32
LR = 5e-5

# ==========================================
# DATASET DEFINITIONS
# ==========================================
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

# ==========================================
# AUGMENTATIONS
# ==========================================
mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
size = MODEL_CONFIG["input_size"][0]

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    AddGaussianNoise(0., 0.02),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ==========================================
    # LOAD DATA
    # ==========================================
    with open(EVAL_DIR / "clean_train_split.json", "r") as f:
        lab_train = json.load(f)
    with open(EVAL_DIR / "clean_val_split.json", "r") as f:
        lab_val = json.load(f)
    with open(EVAL_DIR / "clean_test_split.json", "r") as f:
        lab_test = json.load(f)

    with open(FIELD_SPLITS_PATH, "r") as f:
        field_splits = json.load(f)

    field_trainval_items = []
    for rel_path in field_splits["trainval"]:
        class_name = rel_path.split("/")[0]
        if class_name in CLASS_TO_IDX:
            field_trainval_items.append({
                "path": str(FIELD_DATA_DIR / rel_path),
                "class_name": class_name,
                "class_idx": CLASS_TO_IDX[class_name]
            })

    field_test_items = []
    for rel_path in field_splits["test"]:
        class_name = rel_path.split("/")[0]
        if class_name in CLASS_TO_IDX:
            field_test_items.append({
                "path": str(FIELD_DATA_DIR / rel_path),
                "class_name": class_name,
                "class_idx": CLASS_TO_IDX[class_name]
            })

    random.seed(42)
    random.shuffle(field_trainval_items)
    split_idx = int(len(field_trainval_items) * 0.8)
    field_train = field_trainval_items[:split_idx]
    field_val = field_trainval_items[split_idx:]

    train_data = lab_train + field_train

    print("\n=== DATASET VERIFICATION ===")
    print(f"Lab Train: {len(lab_train)}, Val: {len(lab_val)}, Test: {len(lab_test)}")
    print(f"Field Train: {len(field_train)}, Val: {len(field_val)}, Test: {len(field_test_items)}")
    
    # ==========================================
    # WEIGHTED RANDOM SAMPLER
    # ==========================================
    class_counts = Counter([item['class_idx'] for item in train_data])
    weights = [1.0 / class_counts[item['class_idx']] for item in train_data]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dataset = ExperimentDataset(train_data, transform=train_transforms)
    val_dataset = ExperimentDataset(field_val, transform=eval_transforms)
    test_field_dataset = ExperimentDataset(field_test_items, transform=eval_transforms)
    test_lab_dataset = ExperimentDataset(lab_test, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_field_loader = DataLoader(test_field_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_lab_loader = DataLoader(test_lab_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # LOAD MODEL & FREEZE LAYERS
    # ==========================================
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    print(f"\nLoading BASE Checkpoint: {BASE_WEIGHTS}")
    model.load_state_dict(torch.load(BASE_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE)

    for name, param in model.named_parameters():
        if "features" in name:
            try:
                block_idx = int(name.split(".")[1])
                if block_idx < 9:
                    param.requires_grad = False
            except ValueError:
                param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    def eval_loader(loader):
        if len(loader.dataset) == 0: return 0.0, 0.0
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')

    print("\n=== STARTING TRAINING ===")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)}")
            
        val_acc, val_f1 = eval_loader(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Field Val Acc: {val_acc:.4f} | Field Val Macro F1: {val_f1:.4f}")
        
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), EXP_WEIGHTS)
            print("  -> Saved new best model")

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\n=== FINAL EVALUATION ===")
    model.load_state_dict(torch.load(EXP_WEIGHTS, map_location=DEVICE))
    model.eval()

    def full_eval(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        mac_f1 = f1_score(all_labels, all_preds, average='macro')
        return acc, mac_f1, all_labels, all_preds

    exp_field_acc, exp_field_mac_f1, f_labels, f_preds = full_eval(test_field_loader)
    exp_lab_acc, exp_lab_mac_f1, _, _ = full_eval(test_lab_loader)

    print(f"\nField Accuracy: {exp_field_acc:.4f}")
    print(f"Field Macro F1: {exp_field_mac_f1:.4f}")
    print(f"Lab Accuracy: {exp_lab_acc:.4f}")
    print(f"Lab Macro F1: {exp_lab_mac_f1:.4f}")

if __name__ == '__main__':
    main()
