import os
import zipfile
import shutil
import random
from collections import defaultdict

DESKTOP_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop"
OUTPUT_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset"
MAX_SAMPLES_PER_CLASS = 200 # Limit for fast iteration and balancing

def handle_remove_readonly(func, path, exc):
    import stat
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Ensure output directory exists
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR, onerror=handle_remove_readonly)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = ["plantvillagedataset.zip", "plantdoc.zip", "potato.zip", "ricedisease.zip"]

# Manual mapping rules to merge overlapping classes
CLASS_MAPPING_RULES = {
    "potato_leaf_early_blight": "potato_early_blight",
    "potato_leaf_late_blight": "potato_late_blight",
    "tomato_early_blight_leaf": "tomato_early_blight",
    "tomato_leaf_bacterial_spot": "tomato_bacterial_spot",
    "tomato_leaf_late_blight": "tomato_late_blight",
    "tomato_mold_leaf": "tomato_leaf_mold",
    "tomato_leaf_mosaic_virus": "tomato_mosaic_virus",
    "tomato_tomato_mosaic_virus": "tomato_mosaic_virus",
    "tomato_leaf_yellow_virus": "tomato_yellow_leaf_curl_virus",
    "tomato_tomato_yellowleaf_curl_virus": "tomato_yellow_leaf_curl_virus",
    "bell_pepper_leaf_spot": "pepper_bell_bacterial_spot",
    "bell_pepper_leaf": "pepper_bell_healthy",
    "tomato_leaf": "tomato_healthy",
    # Rice mapping
    "bacterial_leaf_blight": "rice_bacterial_leaf_blight",
    "brown_spot": "rice_brown_spot",
    "healthy_rice_leaf": "rice_healthy",
    "leaf_blast": "rice_leaf_blast",
    "leaf_scald": "rice_leaf_scald",
    "sheath_blight": "rice_sheath_blight",
}

def normalize_class(raw_class):
    cls = raw_class.lower().replace('___', '_').replace('__', '_').replace(' ', '_')
    return CLASS_MAPPING_RULES.get(cls, cls)

def merge_and_extract():
    print(f"Starting extraction to {OUTPUT_DIR}...")
    class_counts = defaultdict(int)

    for dataset_name in DATASETS:
        dataset_path = os.path.join(DESKTOP_DIR, dataset_name)
        if not os.path.exists(dataset_path):
            continue
            
        print(f"Extracting from {dataset_name}...")
        
        with zipfile.ZipFile(dataset_path, 'r') as z:
            file_list = z.infolist()
            # Shuffle so we get a random sample if we hit MAX_SAMPLES
            random.seed(42)
            random.shuffle(file_list)
            
            for file_info in file_list:
                if file_info.is_dir() or not file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                parts = file_info.filename.split('/')
                if len(parts) >= 2:
                    raw_class_name = parts[-2]
                else:
                    continue
                    
                final_class_name = normalize_class(raw_class_name)
                
                # Filter out generic tomato ripe/unripe if they are not diseases
                if final_class_name in ['damaged', 'old', 'ripe', 'unripe']:
                    continue
                
                if class_counts[final_class_name] >= MAX_SAMPLES_PER_CLASS:
                    continue
                    
                class_dir = os.path.join(OUTPUT_DIR, final_class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Extract and rename file to avoid collisions
                ext = file_info.filename.split('.')[-1]
                new_filename = f"{dataset_name.split('.')[0]}_{class_counts[final_class_name]}.{ext}"
                target_path = os.path.join(class_dir, new_filename)
                
                with z.open(file_info) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
                class_counts[final_class_name] += 1

    print("\nExtraction Complete! Unified Class Counts:")
    for cls, count in sorted(class_counts.items()):
        print(f" - {cls}: {count} images")

if __name__ == "__main__":
    merge_and_extract()
