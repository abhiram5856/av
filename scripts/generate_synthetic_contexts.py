"""
Synthetic Context Generator (VLM Wrapper Mock)
==============================================
This script acts as the structural pipeline to process raw JPEGs downloaded from 
the 'raw_telangana' field dataset into the rich JSONL context format required by
the TRACE-RCE v3 training pipeline.

NOTE: Running a Vision-Language Model like LLaVA/GPT-4V over 15,000 images is compute 
intensive. This script demonstrates the structural loop and schema you should use.
"""

import os
import sys
import json
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
RAW_DATA_DIR = REPO_ROOT / "dataset" / "raw_telangana"
OUTPUT_JSONL = REPO_ROOT / "dataset" / "novarcd_contexts.jsonl"

def process_image(image_path: Path, disease_class: str) -> dict:
    """
    Mock function representing the Vision-Language Model extraction.
    In a real scenario, you pass the image to LLaVA or GPT-4o here.
    """
    # MOCK PAYLOAD: Normally returned by VLM
    # This structure matches what novarcd_dataset.py expects to deserialize
    context = {
        "case_id": str(uuid4()),
        "image_path": str(image_path.name),
        "disease_class": disease_class,
        "category": disease_class.split("_")[0].upper(), # e.g. RICE, COTTON
        "ground_truth_causes": [disease_class],
        "context": {
            "system": {"camera_type": "Smartphone", "os_version": "Android", "app_version": "1.0"},
            "user": {"farmer_id": "mock_farmer", "region": "Telangana"},
            "image": {"capture_timestamp": "2026-08-01T12:00:00Z", "lighting_condition": "Sunny"},
            "vision": {
                "predicted_disease": disease_class,
                "confidence_score": 0.88,
                "topk_predictions": {disease_class: 0.88, "healthy": 0.12},
                "concept_activations": {"chlorosis": 0.7}
            },
            "gradcam": {"heatmap_uri": "", "bounding_box": [0,0,100,100]},
            "severity": {"final_severity_index": 45, "affected_area_percentage": 25},
            "weather": {
                "temperature_7d_avg": 31.0,
                "humidity_7d_avg": 75.0,
                "total_precipitation_mm": 12.5,
                "leaf_wetness_hours": 10.0
            },
            "knowledge": {"retrieved_documents": []},
            "history": {
                "total_previous_diagnoses": 2,
                "frequent_crop_diseases": [],
                "last_diagnosis_date": None
            }
        }
    }
    return context

def main():
    print(f"Starting Synthetic Context Generator...")
    
    if not RAW_DATA_DIR.exists():
        print(f"[ERROR] Raw data directory not found: {RAW_DATA_DIR}")
        print("Please run ingest_telangana_field_data.py first!")
        return

    # Count total images
    all_images = list(RAW_DATA_DIR.rglob("*.jpg"))
    print(f"Found {len(all_images)} images across {RAW_DATA_DIR}")

    # Process and append to JSONL
    # (In real run, use 'a' to append, but here we show 'w' for demo purposes)
    processed_count = 0
    with open(OUTPUT_JSONL, "a") as f:
        for img_path in all_images:
            disease_class = img_path.parent.name
            
            # 1. Run VLM extraction (Mocked here)
            record = process_image(img_path, disease_class)
            
            # 2. Write to JSONL
            f.write(json.dumps(record) + "\n")
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(all_images)} images...")
                
    print(f"[SUCCESS] Appended {processed_count} contextual records to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
