import os
import sys
import json
import torch
from pathlib import Path
from torchvision.models import mobilenet_v3_small

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
sys.path.append(str(REPO_ROOT))

from backend.models.class_registry import NUM_CLASSES, CLASS_NAMES

def audit_model_architecture():
    print("--- MODEL ARCHITECTURE AUDIT ---")
    print(f"Registry classes: {NUM_CLASSES}")
    
    ckpt_path = REPO_ROOT / "backend/models/weights/nova_mobilenet_v3_26_classes_init.pth"
    if not ckpt_path.exists():
        ckpt_path = REPO_ROOT / "backend/models/weights/nova_mobilenet_v3.pth"
    if not ckpt_path.exists():
        print(f"FAILED: Checkpoint not found at {ckpt_path}")
        return
        
    try:
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("VERIFIED: Checkpoint successfully loaded into MobileNetV3-Small.")
        print(f"VERIFIED: Checkpoint output dimension matches registry ({NUM_CLASSES}).")
        
    except Exception as e:
        print(f"FAILED: Error loading checkpoint: {e}")

def audit_dataset_splits():
    print("\n--- DATASET LEAKAGE AUDIT ---")
    stats_file = REPO_ROOT / "statistics.json"
    if stats_file.exists():
        with open(stats_file, "r") as f:
            stats = json.load(f)
            print("Found statistics.json")
            for k, v in stats.items():
                print(f"{k}: {v}")
    
    raw_dir = REPO_ROOT / "dataset/raw"
    if raw_dir.exists():
        print("Raw dataset directory found.")
    else:
        print("UNVERIFIED: dataset/raw not found. Cannot independently verify splits.")
        
def audit_metrics():
    print("\n--- METRICS AUDIT ---")
    results_file = REPO_ROOT / "scientific_baseline_results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
            print("Found scientific_baseline_results.json")
            print(f"Reported Accuracy: {results.get('Accuracy', 'N/A')}")
            if "confusion_matrix" in results:
                print("Confusion matrix raw data exists. Metrics can be recomputed.")
            else:
                print("UNVERIFIED: Raw prediction artifacts (confusion matrix) missing.")
    else:
        print("UNVERIFIED: scientific_baseline_results.json missing.")

if __name__ == "__main__":
    audit_model_architecture()
    audit_dataset_splits()
    audit_metrics()
