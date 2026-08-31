import os
import json
import random
from collections import defaultdict

# Add project root to path if needed
import sys
sys.path.append(os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI"))

from backend.models.class_registry import CLASS_NAMES, CLASS_TO_IDX

DATA_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset")
EVAL_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\evaluation")

def create_splits():
    os.makedirs(EVAL_DIR, exist_ok=True)
    audit_path = os.path.join(DATA_DIR, "dataset_audit.json")
    
    if not os.path.exists(audit_path):
        print(f"Audit file not found at {audit_path}. Run audit_dataset.py first.")
        return

    with open(audit_path, "r") as f:
        audit_data = json.load(f)

    hash_dict = audit_data.get("hash_dict", {})
    near_dup_groups = audit_data.get("near_duplicate_groups", [])
    corrupted_files = set(audit_data.get("corrupted_files", []))

    # 1. Build disjoint sets of hashes
    hash_to_group = {}
    group_id_counter = 0

    for near_dup in near_dup_groups:
        group_id_counter += 1
        for h in near_dup:
            hash_to_group[h] = group_id_counter

    # 2. Map every image to a group ID
    # If a hash wasn't in a near-dup group, assign it a new group ID
    path_to_group = {}
    path_to_class = {}
    
    for h, paths in hash_dict.items():
        if h not in hash_to_group:
            group_id_counter += 1
            hash_to_group[h] = group_id_counter
            
        g_id = hash_to_group[h]
        for p in paths:
            if p in corrupted_files:
                continue
            path_to_group[p] = g_id
            
            # Extract class name from path (parent folder)
            class_name = os.path.basename(os.path.dirname(p))
            path_to_class[p] = class_name

    # 3. Aggregate groups
    # group_id -> list of paths
    groups = defaultdict(list)
    for p, g_id in path_to_group.items():
        groups[g_id].append(p)

    # group_id -> majority class
    group_classes = {}
    for g_id, paths in groups.items():
        classes = [path_to_class[p] for p in paths]
        # Get majority class
        majority_class = max(set(classes), key=classes.count)
        group_classes[g_id] = majority_class

    # 4. Stratified Split by Class
    # class -> list of group_ids
    class_to_groups = defaultdict(list)
    for g_id, cls in group_classes.items():
        class_to_groups[cls].append(g_id)

    random.seed(42)

    train_split = []
    val_split = []
    test_split = []

    for cls in CLASS_NAMES:
        g_ids = class_to_groups[cls]
        # Shuffle groups for randomness
        random.shuffle(g_ids)
        
        # Calculate total images in this class
        total_images = sum(len(groups[g]) for g in g_ids)
        
        train_target = int(0.70 * total_images)
        val_target = int(0.15 * total_images)
        
        train_count = 0
        val_count = 0
        
        for g_id in g_ids:
            group_paths = groups[g_id]
            group_size = len(group_paths)
            
            # Formulate records
            records = [
                {
                    "path": p,
                    "class_idx": CLASS_TO_IDX[path_to_class[p]],
                    "class_name": path_to_class[p],
                    "group_id": g_id
                }
                for p in group_paths
            ]
            
            if train_count + group_size <= train_target or (train_count < train_target * 0.9):
                train_split.extend(records)
                train_count += group_size
            elif val_count + group_size <= val_target or (val_count < val_target * 0.9):
                val_split.extend(records)
                val_count += group_size
            else:
                test_split.extend(records)

    # 5. Save splits
    def save_split(split_data, filename):
        path = os.path.join(EVAL_DIR, filename)
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} images to {filename}")

    save_split(train_split, "train_split.json")
    save_split(val_split, "val_split.json")
    save_split(test_split, "test_split.json")
    
    print("Dataset splitting complete. Duplicates and near-duplicates are safely grouped.")

if __name__ == "__main__":
    create_splits()
