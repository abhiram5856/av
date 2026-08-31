"""
Dataset Downloader for Cotton & Turmeric
=========================================
Downloads:
  1. Cotton disease images from HuggingFace (Francesco/cotton-plant-disease)
  2. Turmeric leaf disease images from Mendeley Data (public DOI download)

Organizes them into backend/data/processed_dataset/ with correct class folder names.
"""

import os
import sys
import json
import shutil
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
DATASET_DIR = REPO_ROOT / "backend" / "data" / "processed_dataset"

# ─────────────────────────────────────────────────────────────────────────────
# Target class folder names (must be alphabetically correct)
# ─────────────────────────────────────────────────────────────────────────────
COTTON_CLASSES = {
    "diseased cotton leaf": "cotton_leaf_curl_virus",    # most common disease class
    "fresh cotton leaf":    "cotton_healthy",
    "diseased cotton plant":"cotton_bacterial_blight",
    "fresh cotton plant":   "cotton_healthy",            # merge healthy plant/leaf
    # Alternative mapping if dataset uses these names:
    "bacterial blight":     "cotton_bacterial_blight",
    "curl virus":           "cotton_leaf_curl_virus",
    "grey mildew":          "cotton_grey_mildew",
    "healthy":              "cotton_healthy",
    "fussarium wilt":       "cotton_grey_mildew",        # map to grey mildew if no specific class
}

TURMERIC_CLASSES = {
    "healthy":      "turmeric_healthy",
    "blotch":       "turmeric_leaf_blotch",
    "leaf spot":    "turmeric_leaf_spot",
    "aphids":       "turmeric_leaf_spot",    # merge aphids into leaf spot
}

TARGET_IMAGES_PER_CLASS = 200

def download_cotton_from_hf():
    """Download cotton dataset from HuggingFace."""
    print("\n[COTTON] Downloading from HuggingFace: Francesco/cotton-plant-disease")
    try:
        from datasets import load_dataset
        dataset = load_dataset("Francesco/cotton-plant-disease", split="train", trust_remote_code=True)
        print(f"[COTTON] Loaded {len(dataset)} samples. Label names: {dataset.features['label'].names}")

        label_names = dataset.features['label'].names
        counts = {}

        for i, sample in enumerate(dataset):
            raw_label = label_names[sample['label']].lower().strip()
            # Find matching target class
            target_class = None
            for key, val in COTTON_CLASSES.items():
                if key in raw_label:
                    target_class = val
                    break
            if target_class is None:
                print(f"  [SKIP] Unknown label: {raw_label}")
                continue

            out_dir = DATASET_DIR / target_class
            out_dir.mkdir(parents=True, exist_ok=True)

            existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
            if len(existing) >= TARGET_IMAGES_PER_CLASS:
                continue  # Already have enough

            img = sample['image']
            if isinstance(img, Image.Image):
                fname = out_dir / f"cotton_hf_{i:05d}.jpg"
                img.convert("RGB").save(str(fname), "JPEG", quality=92)
                counts[target_class] = counts.get(target_class, 0) + 1

        print(f"[COTTON] Saved: {counts}")
        return True

    except Exception as e:
        print(f"[COTTON] HuggingFace download failed: {e}")
        return False


def download_cotton_from_github():
    """Fallback: download from a public GitHub release or direct URL."""
    print("\n[COTTON] Trying GitHub fallback (Raghavhari/CottonLeaf_Disease_Detection)...")
    # This specific repo has data accessible via raw.githubusercontent
    # Cotton disease images available via direct URL pattern
    base_urls = {
        "cotton_leaf_curl_virus": [
            "https://raw.githubusercontent.com/Raghavhari/CottonLeaf_Disease_Detection/main/data/Diseased/{i}.jpg"
            for i in range(1, 50)
        ],
    }
    # Note: Most GitHub repos don't host the actual images inline due to size.
    # We rely on HuggingFace primarily.
    print("[COTTON] GitHub direct image download not reliable — HuggingFace is preferred.")
    return False


def download_turmeric_from_mendeley():
    """Download turmeric dataset from Mendeley Data (public access)."""
    print("\n[TURMERIC] Attempting download from Mendeley Data...")
    # Mendeley Data provides direct download links for public datasets
    # Dataset: https://data.mendeley.com/datasets/84p66p6n96/1
    # The ZIP download URL follows this pattern:
    mendeley_url = "https://data.mendeley.com/public-files/datasets/84p66p6n96/files/download"
    
    print("[TURMERIC] Note: Mendeley requires browser authentication for ZIP download.")
    print("[TURMERIC] Trying Roboflow public API instead...")
    return download_turmeric_from_roboflow()


def download_turmeric_from_roboflow():
    """Try Roboflow public export (no login needed for public datasets)."""
    print("[TURMERIC] Searching Roboflow public datasets for turmeric...")
    # Roboflow public datasets can be exported via API
    # https://universe.roboflow.com/ - requires API key for export
    # Instead, we try to find a direct download link
    
    print("[TURMERIC] Roboflow requires API key. Trying HuggingFace search...")
    try:
        from datasets import load_dataset
        # Try loading turmeric dataset if available
        dataset = load_dataset("ttkqwe123/leaf_disease_detection", split="train", trust_remote_code=True)
        print(f"[TURMERIC] Loaded {len(dataset)} samples from ttkqwe123/leaf_disease_detection")
        
        # Filter for turmeric-related samples
        label_names = dataset.features['label'].names if hasattr(dataset.features.get('label', {}), 'names') else []
        print(f"[TURMERIC] Available labels: {label_names[:20]}")
        
        turmeric_saved = {}
        for i, sample in enumerate(dataset):
            raw_label = str(label_names[sample['label']] if 'label' in sample else sample.get('class', '')).lower()
            if 'turmeric' not in raw_label:
                continue
            
            target_class = None
            for key, val in TURMERIC_CLASSES.items():
                if key in raw_label:
                    target_class = val
                    break
            if target_class is None:
                target_class = "turmeric_healthy" if "healthy" in raw_label else "turmeric_leaf_spot"

            out_dir = DATASET_DIR / target_class
            out_dir.mkdir(parents=True, exist_ok=True)
            
            existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
            if len(existing) >= TARGET_IMAGES_PER_CLASS:
                continue

            img = sample.get('image') or sample.get('img')
            if isinstance(img, Image.Image):
                fname = out_dir / f"turmeric_hf_{i:05d}.jpg"
                img.convert("RGB").save(str(fname), "JPEG", quality=92)
                turmeric_saved[target_class] = turmeric_saved.get(target_class, 0) + 1

        print(f"[TURMERIC] Saved from HuggingFace: {turmeric_saved}")
        return bool(turmeric_saved)
        
    except Exception as e:
        print(f"[TURMERIC] HuggingFace attempt failed: {e}")
        return False


def count_current_new_classes():
    new_classes = [
        "cotton_bacterial_blight", "cotton_grey_mildew",
        "cotton_healthy", "cotton_leaf_curl_virus",
        "turmeric_healthy", "turmeric_leaf_blotch", "turmeric_leaf_spot"
    ]
    counts = {}
    for cls in new_classes:
        d = DATASET_DIR / cls
        if d.exists():
            n = len(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")))
            counts[cls] = n
        else:
            counts[cls] = 0
    return counts


def main():
    print("="*60)
    print("AgriVision AI — Cotton & Turmeric Dataset Downloader")
    print("="*60)

    # Step 1: Cotton
    cotton_ok = download_cotton_from_hf()
    if not cotton_ok:
        cotton_ok = download_cotton_from_github()

    # Step 2: Turmeric
    turmeric_ok = download_turmeric_from_mendeley()

    # Step 3: Report
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    counts = count_current_new_classes()
    total_new = 0
    for cls, n in counts.items():
        status = "OK" if n >= 50 else ("PARTIAL" if n > 0 else "EMPTY")
        print(f"  {cls}: {n} images [{status}]")
        total_new += n

    print(f"\nTotal new images downloaded: {total_new}")

    # Save summary
    summary_path = REPO_ROOT / "dataset" / "new_classes_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"new_classes": counts, "total_new_images": total_new}, f, indent=2)
    print(f"Summary saved to {summary_path}")

    if total_new < 100:
        print("\n[WARNING] Not enough images downloaded automatically.")
        print("You will need to manually download from Kaggle.")
        print("Cotton: https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset")
        print("Turmeric: https://www.kaggle.com/datasets/harshvardhan9999/turmeric-leaf-disease-dataset")
    else:
        print("\n[SUCCESS] Dataset ready. Run class registry update and retraining next.")


if __name__ == "__main__":
    main()
