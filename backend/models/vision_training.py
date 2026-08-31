"""
AgriVision AI — Vision Model Training Script
============================================
Architecture:  MobileNetV3-Small (matches saved checkpoint nova_mobilenet_v3.pth)
Dataset:       PlantVillage-Preprocessed-v1.0
Classes:       36 (defined in backend/models/class_registry.py)

IMPORTANT NOTES
---------------
1. Class ordering is determined by torchvision.datasets.ImageFolder, which
   sorts subdirectory names alphabetically.  The class registry in
   class_registry.py is guaranteed to be in the same alphabetical order.
   After training, ALWAYS verify that the discovered class order matches
   class_registry.CLASS_NAMES.

2. This script uses an 80/20 train/val split.  The held-out test set is
   maintained separately in evaluation/test_split.json and must NEVER be
   touched during training or hyperparameter tuning.

3. Training augmentations must NOT be applied during validation or inference.
   The val/inference transform is: Resize(256) → CenterCrop(224) → Normalize.

4. The saved checkpoint contains ONLY the model state_dict (not the full
   model object) to stay framework-version-agnostic.

Reproducing training
--------------------
    python -m backend.models.vision_training

Seed is fixed at 42 for reproducibility.
"""

import os
import copy
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG
from backend.logging.logger import setup_logger

logger = setup_logger("nova.vision_training")


# ─────────────────────────────────────────────────────────────────────────────
# Paths — keep in sync with class_registry.MODEL_CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.normpath(
    r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset"
)
EVAL_DIR = os.path.normpath(
    r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\evaluation"
)
WEIGHTS_DIR = os.path.normpath(
    r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights"
)
CHECKPOINT_FILENAME = MODEL_CONFIG["checkpoint_filename"]   # nova_mobilenet_v3.pth
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss with optional label smoothing
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 reduction: str = "mean", label_smoothing: float = 0.1):
        super().__init__()
        self.alpha           = alpha
        self.gamma           = gamma
        self.reduction       = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss    = F.cross_entropy(inputs, targets, reduction="none",
                                     label_smoothing=self.label_smoothing)
        pt         = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


# ─────────────────────────────────────────────────────────────────────────────
# MixUp augmentation helpers
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, use_cuda: bool = True):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Build MobileNetV3-Small with a custom classification head.

    Architecture choice rationale:
    - Small enough for edge deployment on mobile/Raspberry Pi
    - Achieves ~77% accuracy on PlantVillage 36-class benchmark
    - Checkpoint size ~6 MB — suitable for repository storage
    - Matches the saved checkpoint: nova_mobilenet_v3.pth

    NOTE: If you change architecture here, you MUST retrain and update
    class_registry.MODEL_CONFIG["architecture"] accordingly.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


class JsonDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.records = json.load(f)
        self.transform = transform

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = record["path"]
        label = record["class_idx"]
        
        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for corrupted images if any slipped through
            from PIL import Image
            image = Image.new("RGB", (224, 224))
            
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.records)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(DATA_DIR):
        logger.error(f"Dataset directory not found: {DATA_DIR}")
        return

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ── Transforms ────────────────────────────────────────────────────────
    mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
    size      = MODEL_CONFIG["input_size"][0]  # 224

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Dataset ───────────────────────────────────────────────────────────
    train_split_path = os.path.join(EVAL_DIR, "train_split.json")
    val_split_path = os.path.join(EVAL_DIR, "val_split.json")
    
    if not os.path.exists(train_split_path) or not os.path.exists(val_split_path):
        logger.error(f"Splits not found in {EVAL_DIR}. Run create_splits.py first.")
        return

    train_dataset = JsonDataset(train_split_path, transform=train_transform)
    val_dataset   = JsonDataset(val_split_path, transform=val_transform)

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    logger.info(f"Dataset loaded: {train_size} train, {val_size} val, seed={SEED}")

    # ── Weighted sampler for class imbalance ──────────────────────────────
    train_labels = [r["class_idx"] for r in train_dataset.records]
    class_counts  = Counter(train_labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,  num_workers=0, pin_memory=True)

    # ── Model, loss, optimiser ────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(NUM_CLASSES).to(device)
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    epochs        = 10
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    use_cuda      = device.type == "cuda"

    logger.info(f"Training MobileNetV3-Small on {device} with MixUp (alpha=0.2), FocalLoss")

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        running_loss    = 0.0
        running_correct = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            mixed_x, y_a, y_b, lam = mixup_data(inputs, labels, alpha=0.2, use_cuda=use_cuda)
            outputs = model(mixed_x)
            loss    = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()

            _, preds        = torch.max(outputs, 1)
            running_loss    += loss.item() * inputs.size(0)
            running_correct += (
                lam * preds.eq(y_a).cpu().sum().float()
                + (1 - lam) * preds.eq(y_b).cpu().sum().float()
            )

        scheduler.step()
        train_acc  = running_correct / train_size
        train_loss = running_loss    / train_size

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs        = model(inputs)
                loss           = criterion(outputs, labels)
                _, preds       = torch.max(outputs, 1)
                val_loss      += loss.item() * inputs.size(0)
                val_correct   += torch.sum(preds == labels.data).item()

        epoch_val_acc  = val_correct / val_size
        epoch_val_loss = val_loss    / val_size

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc  = epoch_val_acc
            best_weights  = copy.deepcopy(model.state_dict())
            logger.info(f"  ↑ New best val accuracy: {best_val_acc:.4f}")

    logger.info(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

    # ── Save checkpoint ────────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    checkpoint_path = os.path.join(WEIGHTS_DIR, "best_baseline.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train_model()
