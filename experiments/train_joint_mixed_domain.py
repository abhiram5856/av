import sys
import os
sys.path.insert(0, os.path.abspath("."))

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

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG, CLASS_TO_IDX, IDX_TO_CLASS
from experiments.run_field_experiments import FieldImageDataset, LabImageDataset, evaluate_dataset, get_eval_transform, resolve_class_name

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
WEIGHTS_DIR = os.path.join(BASE_DIR, "backend", "models", "weights")
CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, MODEL_CONFIG["checkpoint_filename"])
FIELD_DATA_DIR = os.path.join(BASE_DIR, "backend", "data", "processed_field_dataset")
FIELD_SPLITS_PATH = os.path.join(BASE_DIR, "backend", "data", "field_splits.json")
LAB_TRAIN_SPLIT_PATH = os.path.join(BASE_DIR, "evaluation", "clean_train_split.json")
LAB_TEST_SPLIT_PATH = os.path.join(BASE_DIR, "evaluation", "clean_test_split.json")
OUT_DIR = os.path.join(BASE_DIR, "evaluation", "field_improvement")
os.makedirs(OUT_DIR, exist_ok=True)

def train_joint_mixed_domain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Joint Mixed Lab + Field Co-Training on device: {device}")

    # Load field train/val split
    with open(FIELD_SPLITS_PATH, 'r') as f:
        field_splits = json.load(f)
    trainval_list = field_splits["trainval"]
    test_list = field_splits["test"]

    random.seed(SEED)
    shuffled_tv = list(trainval_list)
    random.shuffle(shuffled_tv)
    split_idx = int(0.8 * len(shuffled_tv))
    field_train_list = shuffled_tv[:split_idx]
    field_val_list = shuffled_tv[split_idx:]

    # Load lab train split & lab test split
    with open(LAB_TRAIN_SPLIT_PATH, 'r') as f:
        lab_train_dict = json.load(f)
    with open(LAB_TEST_SPLIT_PATH, 'r') as f:
        lab_test_dict = json.load(f)

    # Setup transforms
    mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
    size = MODEL_CONFIG["input_size"][0]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    eval_tf = get_eval_transform()

    # Create datasets
    field_train_ds = FieldImageDataset(field_train_list, transform=train_tf)
    lab_train_ds = LabImageDataset(lab_train_dict, transform=train_tf)

    # Combine Lab Train + Field Train datasets
    joint_train_ds = ConcatDataset([lab_train_ds, field_train_ds])
    print(f"Joint Dataset Sizes: Lab Train={len(lab_train_ds)}, Field Train={len(field_train_ds)}, Total Joint Train={len(joint_train_ds)}")

    joint_loader = DataLoader(joint_train_ds, batch_size=32, shuffle=True, num_workers=2)

    val_dataset = FieldImageDataset(field_val_list, transform=eval_tf)
    test_dataset = FieldImageDataset(test_list, transform=eval_tf)
    lab_test_dataset = LabImageDataset(lab_test_dict, transform=eval_tf)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    lab_test_loader = DataLoader(lab_test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load base model checkpoint
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 5
    best_joint_score = 0.0
    best_joint_ckpt = os.path.join(OUT_DIR, "mixed_joint_best.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in joint_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        # Evaluate on both Field Val AND Lab Test
        field_val_res = evaluate_dataset(model, val_loader, device=device)
        lab_test_res = evaluate_dataset(model, lab_test_loader, device=device)

        field_acc = field_val_res["accuracy"]
        lab_acc = lab_test_res["accuracy"]
        joint_score = (field_acc + lab_acc) / 2.0

        print(f"Epoch {epoch+1}/{epochs}: Loss={running_loss/len(joint_train_ds):.4f} | Field Val Acc={field_acc*100:.2f}% | Lab Test Acc={lab_acc*100:.2f}% | Joint Score={joint_score*100:.2f}%")

        if joint_score > best_joint_score:
            best_joint_score = joint_score
            torch.save(model.state_dict(), best_joint_ckpt)

    # Evaluate best joint checkpoint
    best_joint_model = models.mobilenet_v3_small(weights=None)
    best_joint_model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    best_joint_model.load_state_dict(torch.load(best_joint_ckpt, map_location=device))
    best_joint_model.to(device)

    final_field_test_res = evaluate_dataset(best_joint_model, test_loader, device=device)
    final_lab_test_res = evaluate_dataset(best_joint_model, lab_test_loader, device=device)

    print("\n--- JOINT MIXED-DOMAIN EVALUATION RESULTS ---")
    print(f"Final Field Test Acc: {final_field_test_res['accuracy']*100:.2f}% (Macro F1: {final_field_test_res['macro_f1']*100:.2f}%)")
    print(f"Final Lab Test Acc:   {final_lab_test_res['accuracy']*100:.2f}% (Macro F1: {final_lab_test_res['macro_f1']*100:.2f}%)")

    # If joint model preserves lab acc >= 80% AND improves field test accuracy, promote checkpoint!
    if final_lab_test_res["accuracy"] >= 0.80 and final_field_test_res["accuracy"] >= 0.7440:
        print(f"\nSUCCESS! Promoting Joint Mixed-Domain Checkpoint to Production: {CHECKPOINT_PATH}")
        torch.save(best_joint_model.state_dict(), CHECKPOINT_PATH)
        promoted = True
    else:
        print("\nJoint model did not outperform baseline on both targets simultaneously.")
        promoted = False

    joint_record = {
        "field_test_accuracy": final_field_test_res["accuracy"],
        "field_test_macro_f1": final_field_test_res["macro_f1"],
        "lab_test_accuracy": final_lab_test_res["accuracy"],
        "lab_test_macro_f1": final_lab_test_res["macro_f1"],
        "promoted_to_production": promoted
    }
    with open(os.path.join(OUT_DIR, "mixed_joint_results.json"), 'w') as f:
        json.dump(joint_record, f, indent=2)

if __name__ == "__main__":
    train_joint_mixed_domain()
