import os
import json
from pathlib import Path
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    print("datasets library not installed.")
    exit(1)

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
DATASET_DIR = REPO_ROOT / "backend" / "data" / "processed_dataset"
FIELD_DATASET_DIR = REPO_ROOT / "backend" / "data" / "processed_field_dataset"

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def download_from_agml(dataset_name, class_mapping, crop_prefix):
    print(f"[{crop_prefix.upper()}] Downloading from {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
        label_names = ds.features['label'].names
        
        saved_counts = {v: 0 for v in class_mapping.values()}
        
        for i, item in enumerate(ds):
            raw_label = label_names[item['label']]
            target = class_mapping.get(raw_label)
            
            if target and saved_counts[target] < 150:
                img = item.get('image') or item.get('img')
                if img and isinstance(img, Image.Image):
                    out = DATASET_DIR / target
                    ensure_dir(out)
                    img.convert("RGB").save(str(out / f"{target}_agml_{i}.jpg"), "JPEG", quality=90)
                    saved_counts[target] += 1
                    
        print(f"[{crop_prefix.upper()}] Saved counts: {saved_counts}")
    except Exception as e:
        print(f"[{crop_prefix.upper()}] Failed: {e}")

def main():
    cotton_mapping = {
        "Bacterial_Blight": "cotton_bacterial_blight",
        "Healthy_Leaf": "cotton_healthy"
    }
    download_from_agml("Project-AgML/cotton_leaf_disease_classification", cotton_mapping, "cotton")
    
    groundnut_mapping = {
        "HEALTHY": "groundnut_healthy",
        "RUST": "groundnut_rust",
        "LEAF SPOT": "groundnut_early_leaf_spot",
        "ALTERNARIA LEAF SPOT": "groundnut_late_leaf_spot"
    }
    download_from_agml("Project-AgML/groundnut_leaf_disease_classification", groundnut_mapping, "groundnut")
    
    chilli_mapping = {
        "good": "chilli_healthy"
    }
    download_from_agml("WaruniSHA/chilli-diseases", chilli_mapping, "chilli")

if __name__ == "__main__":
    main()
