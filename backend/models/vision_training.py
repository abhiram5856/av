import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from backend.logging.logger import setup_logger
from collections import Counter

logger = setup_logger("sota_vision_training")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def mixup_data(x, y, alpha=0.2, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for the mixed targets.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def train_sota_model():
    data_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset"
    weights_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights"
    
    if not os.path.exists(data_dir):
        logger.error(f"Dataset directory not found: {data_dir}")
        return
        
    os.makedirs(weights_dir, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(root=data_dir)
    num_classes = len(full_dataset.classes)
    
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            
    train_dataset = TransformSubset(train_ds, transform=train_transform)
    val_dataset = TransformSubset(val_ds, transform=val_transform)
    
    train_labels = [full_dataset.targets[i] for i in train_ds.indices]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    
    criterion = FocalLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    epochs = 10 
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Starting Kaggle-Tier SOTA training on {device} (MixUp enabled)...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # MixUp Implementation
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, use_cuda=torch.cuda.is_available())
            
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()
            
            # Accuracy is harder to calculate with MixUp, we approximate it by picking the dominant label
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (lam * preds.eq(targets_a.data).cpu().sum().float()
                                 + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
            
        scheduler.step()
        
        train_loss = running_loss / train_size
        train_acc = running_corrects.double() / train_size
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        # --- VALIDATION PHASE (No MixUp here) ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_corrects.double() / val_size
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(">>> New Best Model Saved!")
                
    print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.4f}")
    weights_path = os.path.join(weights_dir, "nova_mobilenet_v3.pth") 
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), weights_path)
    logger.info(f"SOTA Model weights saved to {weights_path}")

if __name__ == "__main__":
    train_sota_model()
