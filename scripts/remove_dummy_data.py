"""
Dummy Data Removal Audit
========================
Finds all images containing "dummy" in their filename and removes them.
Generates a manifest of removed files.
"""

import os
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
DOCS_DIR = REPO_ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIRS = [
    REPO_ROOT / "backend" / "data" / "processed_dataset",
    REPO_ROOT / "backend" / "data" / "processed_field_dataset"
]

def main():
    removed_files = []
    
    for d_dir in DATASET_DIRS:
        if not d_dir.exists():
            continue
            
        for cls_dir in d_dir.iterdir():
            if not cls_dir.is_dir():
                continue
                
            for img_path in cls_dir.glob("*dummy*.*"):
                if img_path.is_file():
                    removed_files.append(str(img_path))
                    os.remove(img_path)
                    
    # Generate Manifest
    manifest_path = DOCS_DIR / "DUMMY_DATA_REMOVAL_MANIFEST.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("# Dummy Data Removal Manifest\n\n")
        f.write(f"**Total files removed:** {len(removed_files)}\n\n")
        f.write("## Removed Files\n")
        for file in removed_files:
            # use relative path for readability
            try:
                rel = Path(file).relative_to(REPO_ROOT)
            except:
                rel = file
            f.write(f"- `{rel}`\n")
            
    print(f"Removed {len(removed_files)} dummy images. Manifest written to docs/DUMMY_DATA_REMOVAL_MANIFEST.md")

if __name__ == "__main__":
    main()
