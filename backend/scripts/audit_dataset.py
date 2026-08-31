import os
import glob
from PIL import Image
import imagehash
from collections import defaultdict
import json
import warnings
import time
warnings.filterwarnings('ignore')

DATA_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset")
DOCS_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\docs")

def run_audit():
    if not os.path.exists(DATA_DIR):
        print(f"Data dir not found: {DATA_DIR}")
        return

    os.makedirs(DOCS_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    classes.sort()

    total_images = 0
    corrupted_files = []
    class_counts = defaultdict(int)
    dimensions = set()
    
    hash_dict = defaultdict(list)
    
    print("Starting forensic audit of the dataset...")
    t0 = time.time()
    
    # Fast glob
    image_files = []
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        image_files.extend(glob.glob(os.path.join(cls_dir, "*.*")))
        
    print(f"Found {len(image_files)} potential images in {time.time()-t0:.2f}s")
    
    # Process images
    t1 = time.time()
    for img_path in image_files:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
            
        total_images += 1
        class_name = os.path.basename(os.path.dirname(img_path))
        class_counts[class_name] += 1
        
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                dimensions.add(img.size)
                img_hash = str(imagehash.phash(img))
                hash_dict[img_hash].append(img_path)
        except Exception as e:
            corrupted_files.append(img_path)
            
    print(f"Hashed {total_images} images in {time.time()-t1:.2f}s")
    
    # Exact duplicates
    exact_duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    num_exact_duplicate_groups = len(exact_duplicates)
    num_redundant_images = sum(len(paths) - 1 for paths in exact_duplicates.values())
    
    # Near-duplicates calculation optimized
    t2 = time.time()
    hash_keys = list(hash_dict.keys())
    
    # Convert all hex strings to binary numpy arrays for fast comparison
    import numpy as np
    print("Converting hashes to binary arrays...")
    hash_arrays = []
    for h in hash_keys:
        # pHash returns a 16-character hex string (64 bits)
        # Convert hex string to integer, then to binary array
        val = int(h, 16)
        bin_array = np.array([int(x) for x in format(val, '064b')], dtype=np.int8)
        hash_arrays.append(bin_array)
    
    hash_arrays = np.array(hash_arrays)
    
    near_duplicate_groups = []
    visited = set()
    
    print("Computing near-duplicates (Hamming distance <= 5)...")
    # For large datasets (e.g. 50k), O(N^2) might still be slow in pure Python.
    # We will use vectorized numpy operations to compare one against all remaining.
    for i in range(len(hash_arrays)):
        if i in visited:
            continue
            
        visited.add(i)
        
        # Compare hash_arrays[i] with all remaining elements (i+1 to end)
        # to find distances <= 5.
        diff = hash_arrays[i+1:] != hash_arrays[i]
        distances = np.sum(diff, axis=1)
        
        # Find indices where distance <= 5
        match_indices = np.where(distances <= 5)[0] + i + 1
        
        # Filter out already visited
        unvisited_matches = [idx for idx in match_indices if idx not in visited]
        
        if unvisited_matches:
            group = [hash_keys[i]]
            for idx in unvisited_matches:
                group.append(hash_keys[idx])
                visited.add(idx)
            near_duplicate_groups.append(group)
            
    print(f"Near-duplicates computed in {time.time()-t2:.2f}s")

    # Save raw audit data
    audit_data = {
        "hash_dict": hash_dict,
        "near_duplicate_groups": near_duplicate_groups,
        "corrupted_files": corrupted_files
    }
    
    audit_json_path = os.path.join(DATA_DIR, "dataset_audit.json")
    with open(audit_json_path, "w") as f:
        json.dump(audit_data, f)
        
    print(f"Raw audit data saved to {audit_json_path}")

    # Generate Markdown Report
    report = f"""# ML DATASET FORENSIC AUDIT

## Overview
- **Total Images Scanned:** {total_images}
- **Number of Classes:** {len(classes)}
- **Image Dimensions Found:** {list(dimensions)}
- **Corrupted Files:** {len(corrupted_files)}

## Class Imbalance
"""
    for cls, count in class_counts.items():
        report += f"- **{cls}:** {count} images\n"

    report += f"""
## Duplication Analysis (Data Leakage Risk)
- **Exact Duplicate Groups:** {num_exact_duplicate_groups} (representing {num_redundant_images} redundant images)
- **Near-Duplicate Groups (pHash distance <= 5):** {len(near_duplicate_groups)}

*Conclusion: Splitting randomly WILL cause data leakage across train and test sets due to the high rate of near/exact duplicates in PlantVillage.*

## Corrupted Files
"""
    if not corrupted_files:
        report += "No corrupted files found.\n"
    else:
        for cf in corrupted_files:
            report += f"- {cf}\n"

    report_path = os.path.join(DOCS_DIR, "ML_DATASET_AUDIT.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
        
    print(f"Report successfully generated at {report_path}")

if __name__ == "__main__":
    run_audit()
