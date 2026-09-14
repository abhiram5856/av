import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import math
import cv2
import numpy as np
from PIL import Image
from collections import Counter

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

BASE_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
FIELD_DATA_DIR = os.path.join(BASE_DIR, "backend", "data", "processed_field_dataset")
FIELD_SPLITS_PATH = os.path.join(BASE_DIR, "backend", "data", "field_splits.json")
OUT_DIR = os.path.join(BASE_DIR, "evaluation", "field_improvement")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_ALIAS = {
    "cotton_diseased": "cotton_bacterial_blight",
}

def resolve_class_name(folder_name):
    c = CLASS_ALIAS.get(folder_name, folder_name)
    if c in CLASS_TO_IDX:
        return c
    return None

def analyze_dataset():
    with open(FIELD_SPLITS_PATH, 'r') as f:
        splits = json.load(f)
        
    trainval_paths = splits.get("trainval", [])
    test_paths = splits.get("test", [])
    
    total_images = len(trainval_paths) + len(test_paths)
    
    # Class distribution analysis
    trainval_classes = [resolve_class_name(os.path.dirname(p)) for p in trainval_paths]
    test_classes = [resolve_class_name(os.path.dirname(p)) for p in test_paths]
    
    trainval_counts = Counter(trainval_classes)
    test_counts = Counter(test_classes)
    
    # Image properties sample analysis (sample 200 images from field trainval)
    widths, heights, aspect_ratios = [], [], []
    brightnesses, contrasts, blur_scores = [], [], []
    
    sample_paths = trainval_paths[:200]
    for rel_path in sample_paths:
        full_path = os.path.join(FIELD_DATA_DIR, rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            with Image.open(full_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / max(1, h))
                
            cv_img = cv2.imread(full_path)
            if cv_img is not None:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                brightnesses.append(float(np.mean(gray)))
                contrasts.append(float(np.std(gray)))
                # Laplacian variance as proxy for image sharpness/blur
                blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        except Exception as e:
            pass
            
    analysis_results = {
        "dataset_locations": {
            "field_data_dir": FIELD_DATA_DIR,
            "field_splits_path": FIELD_SPLITS_PATH,
            "checkpoint_path": os.path.join(BASE_DIR, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
        },
        "counts": {
            "total_field_images": total_images,
            "trainval_count": len(trainval_paths),
            "test_count": len(test_paths)
        },
        "class_distribution": {
            "trainval": dict(trainval_counts),
            "test": dict(test_counts)
        },
        "image_statistics_sample": {
            "avg_width": float(np.mean(widths)) if widths else 0.0,
            "avg_height": float(np.mean(heights)) if heights else 0.0,
            "avg_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
            "avg_brightness": float(np.mean(brightnesses)) if brightnesses else 0.0,
            "avg_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
            "avg_blur_score": float(np.mean(blur_scores)) if blur_scores else 0.0
        },
        "domain_observations": {
            "segmented": False,
            "background_complexity": "High (unsegmented soil, weeds, ambient foliage)",
            "lighting_variation": "High (sunlight glare, shadow gradients)",
            "camera_sources": "Multiple smartphone cameras",
            "class_imbalance": "High (Rice & Cotton diseases predominant in field split)"
        }
    }
    
    out_path = os.path.join(OUT_DIR, "forensic_data_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
        
    print(f"Forensic Dataset Analysis completed. Saved to {out_path}")

if __name__ == "__main__":
    analyze_dataset()
