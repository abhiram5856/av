import os
import shutil
from pathlib import Path

# Paths
RAW_DIR = Path("backend/data/raw_field_data")
PROCESSED_DIR = Path("backend/data/processed_field_dataset")

PADDY_DOCTOR_DIR = RAW_DIR / "paddy_doctor" / "train_images"
COTTON_DIR = RAW_DIR / "cotton_disease" / "Cotton Disease" / "train"

# Mappings (raw_folder -> baseline_class)
PADDY_MAPPING = {
    "blast": "rice_leaf_blast",
    "brown_spot": "rice_brown_spot",
    "bacterial_leaf_blight": "rice_bacterial_leaf_blight",
    "normal": "rice_healthy"
}

COTTON_MAPPING = {
    "diseased cotton leaf": "cotton_diseased",
    "fresh cotton leaf": "cotton_healthy"
}

def clear_processed_dir():
    if PROCESSED_DIR.exists():
        print(f"Clearing {PROCESSED_DIR}...")
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True)

def ingest_dataset(source_dir: Path, mapping: dict, dataset_name: str):
    print(f"--- Ingesting {dataset_name} ---")
    if not source_dir.exists():
        print(f"ERROR: {source_dir} not found. Skipping.")
        return

    total_copied = 0
    for raw_folder, target_class in mapping.items():
        src_folder = source_dir / raw_folder
        if not src_folder.exists():
            print(f"  Warning: Expected folder {raw_folder} not found.")
            continue
            
        dst_folder = PROCESSED_DIR / target_class
        dst_folder.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for item in src_folder.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Prefix the filename with the dataset name to avoid collisions
                new_filename = f"{dataset_name.replace(' ', '_').lower()}_{item.name}"
                dst_path = dst_folder / new_filename
                
                # Copy file
                shutil.copy2(item, dst_path)
                count += 1
                total_copied += 1
                
        print(f"  Copied {count} images to {target_class}")
        
    print(f"Total {dataset_name} images ingested: {total_copied}\n")

if __name__ == "__main__":
    clear_processed_dir()
    
    ingest_dataset(PADDY_DOCTOR_DIR, PADDY_MAPPING, "PaddyDoctor")
    ingest_dataset(COTTON_DIR, COTTON_MAPPING, "CottonDisease")
    
    print("Field Data Ingestion Complete!")
