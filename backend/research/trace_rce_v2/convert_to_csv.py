"""
NOVA Root Cause Dataset (NOVA-RCD) — CSV Converter
===================================================
Flattens the generated JSONL AIContext dataset into a tabular CSV file
for easy exploration, analysis, or integration into spreadsheet software.

Output:
  - `dataset/novarcd_summary.csv`
"""

import os
import json
import csv

ROOT_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI"
CONTEXTS_FILE = os.path.join(ROOT_DIR, "dataset", "novarcd_contexts.jsonl")
OUTPUT_CSV = os.path.join(ROOT_DIR, "dataset", "novarcd_summary.csv")

def convert_to_csv():
    print(f"Reading contexts from {CONTEXTS_FILE}...")
    if not os.path.exists(CONTEXTS_FILE):
        raise FileNotFoundError(f"Contexts file not found at {CONTEXTS_FILE}")
        
    records = []
    with open(CONTEXTS_FILE, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Flattening {len(records)} records into tabular structure...")
    
    headers = [
        "case_id", "image_path", "disease_class", "category",
        "vision_confidence", "gradcam_coverage", "severity_score", "severity_level",
        "temperature", "humidity", "rainfall", "leaf_wetness_hours", "wind_speed", "soil_moisture",
        "crop_age_days", "season", "previous_disease",
        "primary_cause", "secondary_cause"
    ]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for r in records:
            ctx = r["context"]
            gt = r["ground_truth_causes"]
            
            primary = gt[0] if len(gt) > 0 else "none"
            secondary = gt[1] if len(gt) > 1 else "none"
            
            # Extract flattened values
            row = [
                r["case_id"],
                r["image_path"],
                r["disease_class"],
                r["category"],
                ctx["vision"]["confidence_score"],
                ctx["gradcam"]["heatmap_coverage_ratio"],
                ctx["severity"]["concern_score"],
                ctx["severity"]["concern_level"],
                ctx["weather"]["temperature_7d_avg"],
                ctx["weather"]["humidity_7d_avg"],
                ctx["weather"]["total_precipitation_mm"],
                ctx["weather"]["leaf_wetness_hours"],
                ctx["weather"]["raw_forecast_summary"].get("wind_speed", 10.0), # fallback if not written directly
                round(1.0 - ctx["severity"]["soil_stress_factor"], 2), # soil moisture
                ctx["history"].get("crop_age_days", 45), # fallback
                "Kharif" if ctx["weather"]["total_precipitation_mm"] > 25.0 else "Rabi",
                ctx["history"].get("previous_disease", "none"),
                primary,
                secondary
            ]
            writer.writerow(row)
            
    print(f"Successfully converted dataset to CSV!")
    print(f"Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    convert_to_csv()
