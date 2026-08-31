"""
NOVA Root Cause Dataset (NOVA-RCD) AIContext Generator
======================================================
This script loads the scientifically justified agricultural metadata annotations
and instantiates the concrete, immutable AIContext schema dataclasses for each image.
It outputs a unified research dataset file in JSONL format.

Output:
  - `dataset/novarcd_contexts.jsonl`: Sealed AIContext entities.
"""

import os
import json
from datetime import datetime

from backend.schemas.context import (
    AIContext, SystemMetadataContext, UserMetadataContext, ImageMetadataContext,
    VisionDetectionContext, GradCAMContext, ConcernContext, WeatherContext,
    RAGKnowledgeContext, PatientHistoryContext
)

ROOT_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI"
ANNOTATIONS_FILE = os.path.join(ROOT_DIR, "dataset", "novarcd_annotations.json")
OUTPUT_JSONL = os.path.join(ROOT_DIR, "dataset", "novarcd_contexts.jsonl")

def build_context(img_path: str, meta: dict) -> AIContext:
    filename = os.path.basename(img_path)
    # Generate unique request id based on hash of image path
    import hashlib
    request_id = "REQ-" + hashlib.md5(img_path.encode()).hexdigest()[:10].upper()
    
    # Extract details
    vis = meta["visual"]
    env = meta["environment"]
    hist = meta["history"]
    sev = meta["severity"]
    knw = meta["knowledge"]
    
    return AIContext(
        system=SystemMetadataContext(
            schema_version="1.0.0",
            environment="research",
            timestamp_utc=datetime.utcnow().isoformat(),
            request_id=request_id
        ),
        user=UserMetadataContext(
            user_id="researcher-01",
            role="agronomist",
            region="Hyderabad-Field-Office",
            preferred_language="en"
        ),
        image=ImageMetadataContext(
            filename=filename,
            width=224,
            height=224,
            format="JPEG",
            aspect_ratio=1.0,
            blur_score=85.0,
            is_valid_quality=True
        ),
        vision=VisionDetectionContext(
            predicted_disease=meta["class_name"],
            confidence_score=vis["vision_confidence"],
            topk_predictions={meta["class_name"]: vis["vision_confidence"], "healthy": 1.0 - vis["vision_confidence"]},
            concept_activations={}
        ),
        gradcam=GradCAMContext(
            heatmap_coverage_ratio=vis["gradcam_coverage"],
            peak_intensity=vis["gradcam_peak"],
            target_layer_name="features.13",
            s3_heatmap_url=f"http://novarcd-s3/{request_id}_heatmap.jpg"
        ),
        severity=ConcernContext(
            base_vision_score=round(sev["severity_score"] * 0.6, 2),
            environmental_risk_factor=round(env["humidity"] / 100.0, 2),
            soil_stress_factor=round(1.0 - env["soil_moisture"], 2),
            concern_score=sev["severity_score"],
            concern_level=sev["severity_level"]
        ),
        weather=WeatherContext(
            latitude=17.385,
            longitude=78.486,
            temperature_7d_avg=env["temperature"],
            humidity_7d_avg=env["humidity"],
            total_precipitation_mm=env["rainfall"],
            leaf_wetness_hours=env["leaf_wetness_hours"],
            raw_forecast_summary={"forecast": f"Average rainfall {env['rainfall']}mm, Temp {env['temperature']}C"}
        ),
        knowledge=RAGKnowledgeContext(
            retrieved_chunk_ids=[f"chunk_{meta['class_name']}_01"],
            document_sources=["pathology_manual_v2.txt"],
            context_text_block=knw["rag_summary"]
        ),
        history=PatientHistoryContext(
            total_previous_diagnoses=5 if meta["category"] != "HEALTHY" else 0,
            frequent_crop_diseases=[meta["class_name"]] if meta["category"] != "HEALTHY" else [],
            last_diagnosis_date="2026-07-01"
        )
    )

def generate_contexts():
    print(f"Loading annotations from {ANNOTATIONS_FILE}...")
    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(f"Annotations file not found at {ANNOTATIONS_FILE}")
        
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
        
    print(f"Generating AIContext structures for {len(annotations)} samples...")
    
    count = 0
    with open(OUTPUT_JSONL, "w") as out:
        for img_path, meta in annotations.items():
            ctx = build_context(img_path, meta)
            
            # Ground truth list (sorted by confidence)
            gt_causes = [c["cause_id"] for c in sorted(meta["ranked_causes"], key=lambda x: x["confidence"], reverse=True)]
            
            # Create serialized record
            record = {
                "case_id": f"NOVA-RCD-{count:05d}",
                "image_path": img_path,
                "disease_class": meta["class_name"],
                "category": meta["category"],
                "context": ctx.to_dict(),
                "ground_truth_causes": gt_causes
            }
            
            out.write(json.dumps(record) + "\n")
            count += 1
            
    print(f"Successfully generated {count} AIContext records!")
    print(f"Output saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    generate_contexts()
