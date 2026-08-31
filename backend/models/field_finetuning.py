import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
from PIL import Image

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG, CLASS_TO_IDX

WEIGHTS_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights")
CHECKPOINT_INIT = os.path.join(WEIGHTS_DIR, "nova_mobilenet_v3_26_classes_init.pth")
CHECKPOINT_TUNED = os.path.join(WEIGHTS_DIR, "nova_mobilenet_v3_field_tuned.pth")

FIELD_DATA_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_field_dataset")
SPLITS_PATH = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\field_splits.json")

SEED = 42
torch.manual_seed(SEED)

class FieldJsonDataset(Dataset):
    def __init__(self, rel_paths, transform=None):
        self.records = []
        for p in rel_paths:
            cls_name = os.path.dirname(p)
            if cls_name in CLASS_TO_IDX:
                self.records.append({
                    "path": os.path.join(FIELD_DATA_DIR, p),
                    "label": CLASS_TO_IDX[cls_name]
                })
        self.transform = transform

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            image = Image.open(record["path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        
        if self.transform:
            image = self.transform(image)
        return image, record["label"]

    def __len__(self):
        return len(self.records)

def build_model_for_finetuning():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    
    # Load the 26-class checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_INIT, map_location="cpu"))
    
    # FREEZE BACKBONE to prevent catastrophic forgetting
    for param in model.features.parameters():
        param.requires_grad = False
        
    return model

def run_field_finetuning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Domain Adaptation on {device}...")
    
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    trainval_paths = splits["trainval"]
    
    # We will just split trainval into 80/20 for internal validation during tuning
    split_idx = int(len(trainval_paths) * 0.8)
    train_paths = trainval_paths[:split_idx]
    val_paths = trainval_paths[split_idx:]
    
    mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
    size = MODEL_CONFIG["input_size"][0]
    
    # HEAVY Domain-Aware Augmentations for field lighting/noise
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_ds = FieldJsonDataset(train_paths, transform=train_transform)
    val_ds = FieldJsonDataset(val_paths, transform=val_transform)
    
    print(f"Field Tuning Dataset: {len(train_ds)} train, {len(val_ds)} val")
    
    # Class weights for sampler
    train_labels = [r["label"] for r in train_ds.records]
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    model = build_model_for_finetuning().to(device)
    
    # Only optimizing the classifier head!
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    epochs = 10
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data).item()
            
        train_acc = running_correct / len(train_ds)
        train_loss = running_loss / len(train_ds)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data).item()
                
        val_acc = val_correct / len(val_ds)
        val_loss = val_loss / len(val_ds)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            
    print(f"Fine-tuning complete. Best Val Acc: {best_val_acc:.4f}")
    
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), CHECKPOINT_TUNED)
    print(f"Tuned checkpoint saved to {CHECKPOINT_TUNED}")

if __name__ == "__main__":
    run_field_finetuning()
