import os
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
DATASET_DIR = REPO_ROOT / "backend" / "data" / "processed_dataset"
FIELD_DATASET_DIR = REPO_ROOT / "backend" / "data" / "processed_field_dataset"

COTTON_CLASSES = [
    "cotton_bacterial_blight",
    "cotton_grey_mildew",
    "cotton_healthy",
    "cotton_leaf_curl_virus"
]

def generate_placeholders():
    total_created = 0
    for cls in COTTON_CLASSES:
        out_dir = DATASET_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir_field = FIELD_DATASET_DIR / cls
        out_dir_field.mkdir(parents=True, exist_ok=True)
        
        # Only generate if empty
        existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
        if len(existing) > 0:
            continue
            
        for i in range(50):
            img = Image.new('RGB', (224, 224), color = (100, 150, 100))
            fname = out_dir / f"{cls}_dummy_{i:03d}.jpg"
            img.save(str(fname))
            if i < 10:
                fname_field = out_dir_field / f"{cls}_dummy_{i:03d}.jpg"
                img.save(str(fname_field))
            total_created += 1
            
    print(f"Created {total_created} placeholder cotton images.")

if __name__ == "__main__":
    generate_placeholders()
