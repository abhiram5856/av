"""
Full Dataset Audit & Clean Split Generator
==========================================
1. Audits entire dataset (both lab and field)
2. Detects exact duplicates via MD5 hashing
3. Creates a stratified grouped split (80/10/10)
4. Ensures duplicate groups are never split across train/val/test
"""

import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = [
    REPO_ROOT / "backend" / "data" / "processed_dataset",
    REPO_ROOT / "backend" / "data" / "processed_field_dataset"
]

def hash_file(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None

def main():
    logger.info("Starting Dataset Audit...")
    
    all_images = []
    class_counts = defaultdict(int)
    
    # 1. Discover all images
    for d_dir in DATASET_DIRS:
        if not d_dir.exists():
            continue
            
        for cls_dir in d_dir.iterdir():
            if not cls_dir.is_dir():
                continue
                
            cls_name = cls_dir.name
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_images.append({
                        "path": str(img_path),
                        "class_name": cls_name
                    })
                    class_counts[cls_name] += 1
                    
    logger.info(f"Discovered {len(all_images)} total images across {len(class_counts)} classes.")
    
    # 2. Duplicate Detection
    logger.info("Computing MD5 hashes for duplicate detection...")
    hash_to_paths = defaultdict(list)
    for img in all_images:
        h = hash_file(img["path"])
        if h:
            hash_to_paths[h].append(img)
            
    # Assign group IDs (duplicate images get the same group ID)
    groups = []
    labels = []
    valid_images = []
    
    group_id_counter = 0
    duplicate_count = 0
    
    for h, imgs in hash_to_paths.items():
        if len(imgs) > 1:
            duplicate_count += (len(imgs) - 1)
        for img in imgs:
            groups.append(group_id_counter)
            labels.append(img["class_name"])
            valid_images.append(img)
        group_id_counter += 1
        
    logger.info(f"Found {duplicate_count} duplicate files.")
    
    # 3. Create class mapping (alphabetical)
    unique_classes = sorted(list(class_counts.keys()))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    # Update valid images with class_idx
    for img in valid_images:
        img["class_idx"] = class_to_idx[img["class_name"]]
        
    # 4. Stratified Group Split
    # We want ~80% Train, 10% Val, 10% Test
    logger.info("Generating grouped stratified splits...")
    
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    # 1 fold for test (10%), 9 folds for train+val
    train_val_idx, test_idx = next(sgkf.split(valid_images, labels, groups))
    
    train_val_images = [valid_images[i] for i in train_val_idx]
    train_val_labels = [labels[i] for i in train_val_idx]
    train_val_groups = [groups[i] for i in train_val_idx]
    
    test_images = [valid_images[i] for i in test_idx]
    
    # Split train_val into train (8/9) and val (1/9)
    sgkf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=42)
    train_idx_inner, val_idx_inner = next(sgkf_val.split(train_val_images, train_val_labels, train_val_groups))
    
    train_images = [train_val_images[i] for i in train_idx_inner]
    val_images = [train_val_images[i] for i in val_idx_inner]
    
    logger.info(f"Split sizes -> Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Save Splits
    with open(EVAL_DIR / "clean_train_split.json", "w") as f:
        json.dump(train_images, f, indent=2)
    with open(EVAL_DIR / "clean_val_split.json", "w") as f:
        json.dump(val_images, f, indent=2)
    with open(EVAL_DIR / "clean_test_split.json", "w") as f:
        json.dump(test_images, f, indent=2)
        
    # Save Report
    report = {
        "total_images": len(all_images),
        "total_classes": len(unique_classes),
        "classes": unique_classes,
        "class_distribution": dict(class_counts),
        "duplicates_found": duplicate_count,
        "splits": {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        },
        "class_mapping": class_to_idx
    }
    with open(EVAL_DIR / "dataset_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Update Class Registry definition so next steps use correct classes
    reg_path = REPO_ROOT / "backend" / "models" / "class_registry.py"
    with open(reg_path, "r", encoding="utf-8") as f:
        reg_lines = f.readlines()
        
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(reg_lines):
        if line.startswith("CLASS_NAMES: List[str] = ["):
            start_idx = i
        elif start_idx != -1 and line.startswith("]"):
            end_idx = i
            break
            
    if start_idx != -1 and end_idx != -1:
        new_classes_str = "CLASS_NAMES: List[str] = [\n"
        for c in unique_classes:
            new_classes_str += f'    "{c}",\n'
        
        reg_lines[start_idx:end_idx] = [new_classes_str]
        
        # update num classes line
        for i, line in enumerate(reg_lines):
            if line.startswith("NUM_CLASSES: int = len(CLASS_NAMES)"):
                reg_lines[i] = f"NUM_CLASSES: int = len(CLASS_NAMES)  # {len(unique_classes)}\n"
                break
                
        with open(reg_path, "w", encoding="utf-8") as f:
            f.writelines(reg_lines)
            
    logger.info("Dataset audit complete. Splits and registry updated.")

if __name__ == "__main__":
    main()
