import os
import subprocess
import sys

DATA_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\raw_field_data")

DATASETS = [
    {"name": "Paddy Doctor", "kaggle_id": "paddy-disease-classification"},
    {"name": "Cotton Disease", "kaggle_id": "janmejaybhoi/cotton-disease-dataset"}
]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Checking Kaggle CLI...")
    try:
        import kaggle
    except ImportError:
        print("Kaggle library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        try:
            import kaggle
        except Exception as e:
            print(f"Failed to install kaggle: {e}")
            return
            
    except Exception as e:
        if "kaggle.json" in str(e).lower() or "unauthorized" in str(e).lower():
            print("============================================================")
            print("KAGGLE API UNAUTHORIZED / NOT CONFIGURED")
            print("============================================================")
            print("To automate downloading the field datasets, you need a kaggle.json file.")
            print("1. Go to kaggle.com -> Account -> Create New API Token")
            print("2. Place kaggle.json in C:\\Users\\ABHIRAM MODUKURU\\.kaggle\\kaggle.json")
            print("\nAlternatively, manually download and extract these to backend/data/raw_field_data:")
            print("- https://www.kaggle.com/competitions/paddy-disease-classification/data")
            print("- https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset")
            print("- https://github.com/pratikkayal/PlantDoc-Dataset")
            print("============================================================")
            return
            
    print("Kaggle API configured. Attempting to download datasets...")
    
    for ds in DATASETS:
        dest_folder = os.path.join(DATA_DIR, ds['name'].lower().replace(' ', '_'))
        os.makedirs(dest_folder, exist_ok=True)
        print(f"Downloading {ds['name']}...")
        try:
            if "competitions" in ds['kaggle_id'] or "/" not in ds['kaggle_id']:
                subprocess.check_call(["kaggle", "competitions", "download", "-c", ds['kaggle_id'], "-p", dest_folder])
            else:
                subprocess.check_call(["kaggle", "datasets", "download", "-d", ds['kaggle_id'], "-p", dest_folder])
                
            print(f"Successfully downloaded {ds['name']}. Please extract the zip files.")
        except Exception as e:
            print(f"Failed to download {ds['name']}. Error: {e}")
            print("Please ensure you have accepted the competition rules on Kaggle for Paddy Doctor.")

if __name__ == "__main__":
    main()
