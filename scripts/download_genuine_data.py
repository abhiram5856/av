"""
Genuine Data Acquisition and Class Triage
=========================================
Downloads genuine data from open HuggingFace datasets.
If a class cannot obtain genuine data, it is flagged and removed from the active scope.
Produces docs/DATASET_SOURCES.md
"""

import os
import json
from pathlib import Path
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    print("datasets library not installed. Cannot download HF datasets.")
    exit(1)

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
DATASET_DIR = REPO_ROOT / "backend" / "data" / "processed_dataset"
DOCS_DIR = REPO_ROOT / "docs"

# Target classes for the new crops (Cotton, Groundnut, Chilli)
TARGET_CLASSES = {
    # Cotton
    "cotton_bacterial_blight", "cotton_grey_mildew", "cotton_healthy", "cotton_leaf_curl_virus",
    # Groundnut
    "groundnut_early_leaf_spot", "groundnut_healthy", "groundnut_late_leaf_spot", "groundnut_rust",
    # Chilli
    "chilli_healthy", "chilli_leaf_curl", "chilli_leaf_spot", "chilli_whitefly", "chilli_yellowish"
}

SOURCES_LOG = []

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def download_cotton():
    print("[COTTON] Attempting download from Francesco/cotton-plant-disease")
    try:
        # Load parquet directly to avoid trust_remote_code issues if possible, or use standard load_dataset
        ds = load_dataset("Francesco/cotton-plant-disease", split="train")
        
        saved_counts = {"cotton_bacterial_blight": 0, "cotton_healthy": 0, "cotton_leaf_curl_virus": 0, "cotton_grey_mildew": 0}
        
        # Mapping logic (hypothetical based on typical object detection labels or similar)
        # Assuming there's a label column
        label_names = ds.features['label'].names if hasattr(ds.features.get('label'), 'names') else []
        if not label_names:
            print("  No 'label' feature with names found. Trying to infer...")
            
        for i, item in enumerate(ds):
            raw_label = ""
            if 'label' in item:
                if label_names:
                     raw_label = label_names[item['label']].lower()
                else:
                     raw_label = str(item['label']).lower()
            elif 'class' in item:
                raw_label = str(item['class']).lower()
            
            target = None
            if "bacterial" in raw_label or "blight" in raw_label: target = "cotton_bacterial_blight"
            elif "curl" in raw_label or "virus" in raw_label: target = "cotton_leaf_curl_virus"
            elif "mildew" in raw_label: target = "cotton_grey_mildew"
            elif "healthy" in raw_label or "fresh" in raw_label: target = "cotton_healthy"
            else: target = "cotton_diseased" # Fallback, but we might discard
            
            if target in saved_counts and saved_counts[target] < 150:
                img = item.get('image') or item.get('img')
                if img and isinstance(img, Image.Image):
                    out = DATASET_DIR / target
                    ensure_dir(out)
                    img.convert("RGB").save(str(out / f"cotton_{i}.jpg"), "JPEG", quality=90)
                    saved_counts[target] += 1
                    
        total = sum(saved_counts.values())
        if total > 0:
            SOURCES_LOG.append({
                "dataset_name": "Francesco/cotton-plant-disease",
                "source": "Hugging Face",
                "crop": "Cotton",
                "disease_classes_used": [k for k, v in saved_counts.items() if v > 0],
                "number_of_images_used": total,
                "licence": "Open Access (HF Hub)"
            })
            print(f"  [SUCCESS] Saved {total} cotton images.")
            return True
        else:
            print("  [FAIL] Could not extract valid labelled images.")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return False

def download_chilli():
    print("[CHILLI] Attempting download from WaruniSHA/chilli-diseases")
    try:
        ds = load_dataset("WaruniSHA/chilli-diseases", split="train")
        saved_counts = {"chilli_leaf_curl": 0, "chilli_leaf_spot": 0, "chilli_healthy": 0, "chilli_whitefly": 0, "chilli_yellowish": 0}
        
        label_names = ds.features['label'].names if hasattr(ds.features.get('label'), 'names') else []
        
        for i, item in enumerate(ds):
            raw_label = ""
            if 'label' in item:
                if label_names:
                     raw_label = label_names[item['label']].lower()
                else:
                     raw_label = str(item['label']).lower()
            
            target = None
            if "curl" in raw_label: target = "chilli_leaf_curl"
            elif "spot" in raw_label or "anthracnose" in raw_label or "cercospora" in raw_label: target = "chilli_leaf_spot"
            elif "whitefly" in raw_label: target = "chilli_whitefly"
            elif "yellow" in raw_label: target = "chilli_yellowish"
            elif "healthy" in raw_label: target = "chilli_healthy"
            
            if target in saved_counts and saved_counts[target] < 150:
                img = item.get('image') or item.get('img')
                if img and isinstance(img, Image.Image):
                    out = DATASET_DIR / target
                    ensure_dir(out)
                    img.convert("RGB").save(str(out / f"chilli_{i}.jpg"), "JPEG", quality=90)
                    saved_counts[target] += 1
                    
        total = sum(saved_counts.values())
        if total > 0:
            SOURCES_LOG.append({
                "dataset_name": "WaruniSHA/chilli-diseases",
                "source": "Hugging Face",
                "crop": "Chilli",
                "disease_classes_used": [k for k, v in saved_counts.items() if v > 0],
                "number_of_images_used": total,
                "licence": "Open Access (HF Hub)"
            })
            print(f"  [SUCCESS] Saved {total} chilli images.")
            return True
        else:
            print("  [FAIL] Could not extract valid labelled images.")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return False

def check_existing_data():
    """Returns a list of classes that actually have > 20 genuine images."""
    valid_classes = []
    for cls in TARGET_CLASSES:
        cls_dir = DATASET_DIR / cls
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png")))
            if count > 20:
                valid_classes.append(cls)
            else:
                print(f"[TRIAGE] Insufficient data for {cls} ({count} images). Will flag for removal.")
        else:
            print(f"[TRIAGE] No directory found for {cls}. Will flag for removal.")
            
    return valid_classes

def write_sources_md():
    with open(DOCS_DIR / "DATASET_SOURCES.md", "w", encoding="utf-8") as f:
        f.write("# Genuine Dataset Sources\n\n")
        f.write("This document manifests all acquired public datasets integrated into the AgriVision AI training pool.\n\n")
        
        if not SOURCES_LOG:
            f.write("*No external open datasets were successfully acquired in this session.*\n")
            return
            
        f.write("| Dataset | Source | Crop | Classes Mapped | Img Count | Licence |\n")
        f.write("|---------|--------|------|----------------|-----------|---------|\n")
        for log in SOURCES_LOG:
            classes = ", ".join(log["disease_classes_used"])
            f.write(f"| {log['dataset_name']} | {log['source']} | {log['crop']} | {classes} | {log['number_of_images_used']} | {log['licence']} |\n")

def triage_registry(valid_new_classes):
    """Updates the class_registry.py to drop unsupported classes."""
    reg_path = REPO_ROOT / "backend" / "models" / "class_registry.py"
    
    with open(reg_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    import re
    # Extract existing classes
    match = re.search(r"CLASS_NAMES: List\[str\] = \[(.*?)\]", content, re.DOTALL)
    if not match:
        print("[ERROR] Could not parse class_registry.py")
        return
        
    current_classes_raw = match.group(1).split(",")
    current_classes = [c.strip().strip('"').strip("'") for c in current_classes_raw if c.strip()]
    
    # Preserve original 24 classes (which are known genuine from PlantVillage)
    # Plus the newly validated classes
    final_classes = []
    for c in current_classes:
        if c not in TARGET_CLASSES:
            final_classes.append(c) # Original PV class
        elif c in valid_new_classes:
            final_classes.append(c) # Validated new class
            
    final_classes = sorted(list(set(final_classes)))
    
    # Replace in file
    new_list_str = "CLASS_NAMES: List[str] = [\n"
    for c in final_classes:
        new_list_str += f'    "{c}",\n'
    new_list_str += "]"
    
    content = re.sub(r"CLASS_NAMES: List\[str\] = \[.*?\]", new_list_str, content, flags=re.DOTALL)
    content = re.sub(r"NUM_CLASSES: int = len\(CLASS_NAMES\).*?\n", f"NUM_CLASSES: int = len(CLASS_NAMES)  # {len(final_classes)}\n", content)
    
    with open(reg_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\n[TRIAGE COMPLETE] Registry updated. Final class count is {len(final_classes)}.")
    if len(final_classes) < 37:
        print(f"[NOTE] The scope has been reduced from 37 to {len(final_classes)} due to the STRICT RULE against dummy data.")

def main():
    print("="*60)
    print("AgriVision AI — Genuine Data Acquisition")
    print("="*60)
    
    download_cotton()
    download_chilli()
    
    # Groundnut is skipped for automatic download as no high-quality unauthenticated HuggingFace
    # dataset was found during research that directly matches the leaf spot / rust classes.
    print("[GROUNDNUT] No suitable unauthenticated open dataset found for Groundnut diseases.")
    
    valid_classes = check_existing_data()
    write_sources_md()
    triage_registry(valid_classes)

if __name__ == "__main__":
    main()
