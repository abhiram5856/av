import os
import zipfile
import tarfile
import json
from collections import defaultdict

DESKTOP_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop"
OUTPUT_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = [
    "plantvillagedataset.zip",
    "plantdoc.zip",
    "potato.zip",
    "ricedisease.zip",
    "tomato.zip"
]

def scan_datasets():
    class_mapping = defaultdict(lambda: defaultdict(int))
    total_images = 0

    for dataset_name in DATASETS:
        dataset_path = os.path.join(DESKTOP_DIR, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Skipping {dataset_name} (Not found on Desktop)")
            continue
            
        print(f"Scanning {dataset_name}...")
        
        try:
            if dataset_path.endswith('.zip'):
                with zipfile.ZipFile(dataset_path, 'r') as z:
                    for file_info in z.infolist():
                        # Skip directories
                        if file_info.is_dir():
                            continue
                        
                        # Only count image files
                        if not file_info.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
                            continue
                            
                        # Extract the immediate parent directory name as the class name
                        # Examples:
                        # PlantVillage/Tomato___Early_blight/image1.jpg -> Tomato___Early_blight
                        # potato/early_blight/1.jpg -> early_blight
                        parts = file_info.filename.split('/')
                        if len(parts) >= 2:
                            class_name = parts[-2]
                        else:
                            class_name = "unknown"
                            
                        # Normalize class name for easier merging later
                        normalized_class = class_name.lower().replace('___', '_').replace('__', '_').replace(' ', '_')
                        
                        class_mapping[normalized_class][dataset_name] += 1
                        total_images += 1
                        
        except Exception as e:
            print(f"Error reading {dataset_name}: {e}")

    # Save the raw mapping to analyze overlapping classes
    output_path = os.path.join(OUTPUT_DIR, 'dataset_class_mapping.json')
    with open(output_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
        
    print(f"\nScan complete! Found {total_images} total images across {len(class_mapping)} unique (normalized) classes.")
    print(f"Mapping saved to {output_path}")

if __name__ == "__main__":
    scan_datasets()
