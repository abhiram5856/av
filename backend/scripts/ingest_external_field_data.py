import os
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXTERNAL_DIR = REPO_ROOT / "backend" / "data" / "external_field"
REGISTRY_PATH = REPO_ROOT / "evaluation" / "external_field_data_registry.json"

# We must NEVER let external data overlap with existing validation/test splits.
EXISTING_SPLITS = [
    REPO_ROOT / "backend" / "data" / "field_splits.json",
    REPO_ROOT / "evaluation" / "clean_test_split.json",
    REPO_ROOT / "evaluation" / "clean_train_split.json"
]

def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def build_existing_hash_registry() -> set:
    print("Building SHA256 registry of all existing images to prevent data leakage...")
    existing_hashes = set()
    
    # 1. Load Field Splits
    if EXISTING_SPLITS[0].exists():
        with open(EXISTING_SPLITS[0]) as f:
            field_splits = json.load(f)
            # Add all test, val, and train images to the hash registry
            for split_name, paths in field_splits.items():
                for rel_path in paths:
                    full_path = REPO_ROOT / "backend" / "data" / "processed_field_dataset" / rel_path
                    if full_path.exists():
                        existing_hashes.add(calculate_sha256(str(full_path)))
                        
    # 2. Load Lab Splits
    for lab_split in EXISTING_SPLITS[1:]:
        if lab_split.exists():
            with open(lab_split) as f:
                data = json.load(f)
                paths = []
                if isinstance(data, dict):
                    for v in data.values(): paths.extend(v)
                else:
                    paths = [x['path'] for x in data]
                
                for rel_path in paths:
                    full_path = REPO_ROOT / rel_path
                    if full_path.exists():
                        existing_hashes.add(calculate_sha256(str(full_path)))
                        
    print(f"Registered {len(existing_hashes)} existing unique image hashes.")
    return existing_hashes

def ingest_dataset(dataset_id: str, mappings: Dict[str, dict], existing_hashes: set):
    """
    Ingests a raw downloaded dataset, checks for duplicates, and maps it strictly 
    to the canonical taxonomy.
    """
    dataset_dir = EXTERNAL_DIR / dataset_id
    if not dataset_dir.exists():
        print(f"[{dataset_id}] Raw directory not found. Skipping.")
        return

    output_dir = EXTERNAL_DIR / f"{dataset_id}_cleaned"
    output_dir.mkdir(exist_ok=True)
    
    stats = {
        "original_count": 0,
        "exact_matches_saved": 0,
        "ambiguous_rejected": 0,
        "duplicates_rejected": 0,
        "corrupted_rejected": 0
    }

    # Walk through the raw dataset
    for root, _, files in os.walk(dataset_dir):
        raw_class_name = os.path.basename(root)
        
        # Check if the class is in our strict JSON mapping
        if raw_class_name not in mappings:
            continue
            
        mapping_info = mappings[raw_class_name]
        
        if mapping_info["status"] != "EXACT MATCH":
            stats["ambiguous_rejected"] += len(files)
            continue
            
        canonical_class = mapping_info["canonical"]
        class_out_dir = output_dir / canonical_class
        class_out_dir.mkdir(exist_ok=True)
        
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
                
            stats["original_count"] += 1
            file_path = os.path.join(root, file)
            
            # 1. Verify readability
            try:
                # Basic check, would use cv2 or PIL in production
                if os.path.getsize(file_path) < 1024:
                    raise ValueError("File too small")
            except Exception:
                stats["corrupted_rejected"] += 1
                continue
                
            # 2. Duplicate Detection
            file_hash = calculate_sha256(file_path)
            if file_hash in existing_hashes:
                stats["duplicates_rejected"] += 1
                continue
                
            # 3. Save
            shutil.copy2(file_path, class_out_dir / file)
            existing_hashes.add(file_hash)
            stats["exact_matches_saved"] += 1

    print(f"\n--- Ingestion Report for {dataset_id} ---")
    print(f"Original Images Scanned: {stats['original_count']}")
    print(f"Successfully Cleaned & Mapped: {stats['exact_matches_saved']}")
    print(f"Rejected (Ambiguous Class): {stats['ambiguous_rejected']}")
    print(f"Rejected (Test Leak/Duplicate): {stats['duplicates_rejected']}")
    print(f"Rejected (Corrupted): {stats['corrupted_rejected']}")

def main():
    print("Starting Strict External Data Ingestion Pipeline...\n")
    if not REGISTRY_PATH.exists():
        print(f"Fatal: Registry missing at {REGISTRY_PATH}")
        return
        
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
        
    existing_hashes = build_existing_hash_registry()
    
    for dataset in registry.get("datasets", []):
        dataset_id = dataset["dataset_id"]
        mappings = dataset["canonical_mappings"]
        ingest_dataset(dataset_id, mappings, existing_hashes)
        
    print("\nIngestion pipeline complete. Clean datasets are ready in *_cleaned directories.")
    print("DO NOT train the model until per-class balancing limits are set.")

if __name__ == "__main__":
    main()
