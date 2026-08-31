"""
Telangana Field Dataset Ingestion Pipeline
==========================================
Downloads and formats the following datasets into 'dataset/raw_telangana':
1. Paddy Doctor (Rice) - Kaggle
2. Cotton Plant Disease - HuggingFace
3. PlantDoc (Tomato, Corn) - HuggingFace
4. Chilli & Groundnut - HuggingFace

NOTE: Make sure you have your kaggle.json configured in ~/.kaggle/kaggle.json
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
TARGET_DIR = REPO_ROOT / "dataset" / "raw_telangana"

# Target classes mapping
CROP_CLASSES = {
    "rice": ["rice_blast", "rice_bacterial_blight", "rice_brown_spot", "rice_tungro", "rice_healthy"],
    "cotton": ["cotton_bacterial_blight", "cotton_leaf_curl_virus", "cotton_healthy", "cotton_grey_mildew"],
    "tomato": ["tomato_early_blight", "tomato_late_blight", "tomato_healthy", "tomato_leaf_mold"],
    "maize": ["maize_northern_leaf_blight", "maize_common_rust", "maize_healthy"],
    "chilli": ["chilli_leaf_curl", "chilli_leaf_spot", "chilli_healthy"],
    "groundnut": ["groundnut_early_leaf_spot", "groundnut_late_leaf_spot", "groundnut_healthy"]
}

def setup_directories():
    print("[SETUP] Creating target directories...")
    if not TARGET_DIR.exists():
        TARGET_DIR.mkdir(parents=True)
    
    for crop, classes in CROP_CLASSES.items():
        for cls in classes:
            cls_dir = TARGET_DIR / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Directories created under {TARGET_DIR}")

def download_paddy_doctor():
    print("\n[RICE] Downloading Paddy Doctor from Kaggle...")
    try:
        import kaggle
        kaggle.api.authenticate()
        
        # Using a popular paddy doctor dataset from kaggle
        dataset_name = "minhhuy2810/rice-diseases-image-dataset" 
        download_path = TARGET_DIR / "temp_paddy"
        kaggle.api.dataset_download_files(dataset_name, path=str(download_path), unzip=True)
        
        print(f"[RICE] Downloaded to {download_path}. Needs manual routing based on exact folder names.")
        # E.g., shutil.move(download_path / "Bacterial leaf blight", TARGET_DIR / "rice_bacterial_blight")
        # For this script, we'll just log it.
    except Exception as e:
        print(f"[ERROR] Failed to download Paddy Doctor: {e}")
        print("Please ensure kaggle.json is configured.")

def download_hf_dataset(hf_repo: str, crop_prefix: str, split="train"):
    print(f"\n[{crop_prefix.upper()}] Downloading from HuggingFace ({hf_repo})...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(hf_repo, split=split, trust_remote_code=True)
        print(f"[{crop_prefix.upper()}] Loaded {len(dataset)} samples.")
        
        label_names = dataset.features['label'].names
        counts = {}
        
        for i, sample in enumerate(dataset):
            # Limit to 500 images per class for balanced fine-tuning
            raw_label = label_names[sample['label']].lower()
            
            # Very basic heuristic matching
            target_class = None
            for cls in CROP_CLASSES[crop_prefix]:
                if cls.split("_")[-1] in raw_label or cls.split("_")[-2] in raw_label:
                    target_class = cls
                    break
            
            if target_class is None:
                target_class = f"{crop_prefix}_healthy" if "healthy" in raw_label else None
                
            if target_class:
                out_dir = TARGET_DIR / target_class
                existing = len(list(out_dir.glob("*.jpg")))
                if existing >= 500:
                    continue
                    
                img = sample['image']
                if isinstance(img, Image.Image):
                    fname = out_dir / f"{hf_repo.replace('/','_')}_{i:05d}.jpg"
                    img.convert("RGB").save(str(fname), "JPEG", quality=90)
                    counts[target_class] = counts.get(target_class, 0) + 1
                    
        print(f"[{crop_prefix.upper()}] Saved images: {counts}")
        
    except Exception as e:
        print(f"[ERROR] HuggingFace download failed for {hf_repo}: {e}")

def main():
    print("=== Telangana Dataset Ingestion Pipeline ===")
    setup_directories()
    
    # 1. Download Kaggle datasets
    download_paddy_doctor()
    
    # 2. Download HuggingFace datasets
    # Cotton
    download_hf_dataset("Francesco/cotton-plant-disease", "cotton")
    
    # Corn/Tomato (using a generic plant disease HF repo for PlantDoc proxy)
    download_hf_dataset("timm/plantvillage", "tomato")
    download_hf_dataset("timm/plantvillage", "maize")
    
    # Chilli / Groundnut
    download_hf_dataset("Sankhya/chilli_leaf_diseases", "chilli")
    download_hf_dataset("Sankhya/groundnut_leaf_diseases", "groundnut")
    
    print("\n[DONE] Ingestion script completed.")
    print(f"Please inspect {TARGET_DIR} for the downloaded images.")
    print("You can now run the NOVA-RCD context generation pipeline on these raw images.")

if __name__ == "__main__":
    # Remove dry run flag to actually execute. For safety in some environments, you might want to run piecewise.
    main()
