import os
import glob
from PIL import Image
import imagehash
from collections import defaultdict
import json
import warnings
import time
import numpy as np
import random

warnings.filterwarnings('ignore')

FIELD_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_field_dataset")
LAB_AUDIT_PATH = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\dataset_audit.json")
SPLIT_OUTPUT = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\field_splits.json")

def hash_to_binary_array(hex_hash):
    val = int(hex_hash, 16)
    return np.array([int(x) for x in format(val, '064b')], dtype=np.int8)

def run_field_audit():
    print("Loading Lab Baseline Hashes...")
    with open(LAB_AUDIT_PATH, 'r') as f:
        lab_audit = json.load(f)
    lab_hashes = list(lab_audit["hash_dict"].keys())
    lab_hash_arrays = np.array([hash_to_binary_array(h) for h in lab_hashes])
    
    classes = sorted([d for d in os.listdir(FIELD_DIR) if os.path.isdir(os.path.join(FIELD_DIR, d))])
    
    image_files = []
    for cls in classes:
        image_files.extend(glob.glob(os.path.join(FIELD_DIR, cls, "*.*")))
        
    print(f"Found {len(image_files)} field images. Computing hashes...")
    
    field_hashes = {}
    corrupted = []
    
    t0 = time.time()
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                h = str(imagehash.phash(img))
                if h not in field_hashes:
                    field_hashes[h] = []
                field_hashes[h].append(img_path)
        except Exception:
            corrupted.append(img_path)
            
    print(f"Hashed {len(image_files)} field images in {time.time()-t0:.2f}s")
    print(f"Unique field hashes: {len(field_hashes)}")
    
    # 1. Leakage Verification against Lab Dataset
    print("\n--- Cross-Domain Leakage Check ---")
    t1 = time.time()
    field_hash_keys = list(field_hashes.keys())
    field_hash_arrays = np.array([hash_to_binary_array(h) for h in field_hash_keys])
    
    leaked_hashes = set()
    for i in range(len(field_hash_arrays)):
        diff = lab_hash_arrays != field_hash_arrays[i]
        distances = np.sum(diff, axis=1)
        if np.any(distances <= 5):
            leaked_hashes.add(field_hash_keys[i])
            
    print(f"Found {len(leaked_hashes)} field images that leaked from the lab dataset in {time.time()-t1:.2f}s.")
    
    safe_field_hashes = [h for h in field_hash_keys if h not in leaked_hashes]
    safe_field_hash_arrays = np.array([hash_to_binary_array(h) for h in safe_field_hashes])
    
    # 2. Internal Near-Duplicate Grouping
    print("\n--- Internal Field Grouping ---")
    visited = set()
    near_duplicate_groups = []
    
    for i in range(len(safe_field_hash_arrays)):
        if i in visited:
            continue
        visited.add(i)
        
        diff = safe_field_hash_arrays[i+1:] != safe_field_hash_arrays[i]
        distances = np.sum(diff, axis=1)
        match_indices = np.where(distances <= 5)[0] + i + 1
        
        unvisited = [idx for idx in match_indices if idx not in visited]
        group = [safe_field_hashes[i]]
        
        for idx in unvisited:
            group.append(safe_field_hashes[idx])
            visited.add(idx)
            
        near_duplicate_groups.append(group)
        
    print(f"Grouped into {len(near_duplicate_groups)} safe isolated groups.")
    
    # 3. Splitting logic (TrainVal vs Test)
    print("\n--- Generating Duplicate-Safe Splits ---")
    
    # We want 15% Test. We split by groups.
    random.seed(42)
    random.shuffle(near_duplicate_groups)
    
    total_safe_images = sum(len(field_hashes[h]) for group in near_duplicate_groups for h in group)
    test_target = int(total_safe_images * 0.15)
    
    test_paths = []
    trainval_paths = []
    
    for group in near_duplicate_groups:
        group_paths = []
        for h in group:
            # We store relative paths starting from class folder
            for p in field_hashes[h]:
                cls_name = os.path.basename(os.path.dirname(p))
                img_name = os.path.basename(p)
                group_paths.append(f"{cls_name}/{img_name}")
                
        if len(test_paths) < test_target:
            test_paths.extend(group_paths)
        else:
            trainval_paths.extend(group_paths)
            
    print(f"Train/Val: {len(trainval_paths)} images")
    print(f"Test: {len(test_paths)} images")
    
    splits = {
        "trainval": trainval_paths,
        "test": test_paths,
        "leaked": sum(len(field_hashes[h]) for h in leaked_hashes),
        "corrupted": len(corrupted)
    }
    
    with open(SPLIT_OUTPUT, 'w') as f:
        json.dump(splits, f, indent=2)
        
    print(f"Splits saved to {SPLIT_OUTPUT}")

if __name__ == "__main__":
    run_field_audit()
