import os
import urllib.request
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO)

DATASETS = {
    "tomato_pakistan": "https://data.mendeley.com/datasets/3mbnb82mxd/1/download",
    "potato_pldd_up": "https://data.mendeley.com/datasets/3j4nfkvp2n/1/download",
    "groundnut_karnataka": "https://data.mendeley.com/datasets/22p2vcbxfk/3/download",
    "chilli_india": "https://data.mendeley.com/datasets/ymt8k9bjkn/1/download",
    "maize_field": "https://researchdata.up.ac.za/api/v2/articles/20237613/download"
}

BASE_DIR = os.path.join("backend", "data", "external_field")

def download_datasets():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    for name, url in DATASETS.items():
        dataset_dir = os.path.join(BASE_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)
        zip_path = os.path.join(dataset_dir, f"{name}.zip")
        
        logging.info(f"Attempting to download {name} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            logging.info(f"Successfully downloaded {name}. Extracting...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            logging.info(f"Extracted {name}.")
            os.remove(zip_path)
            
        except Exception as e:
            logging.error(f"Failed to download/extract {name}: {e}")

if __name__ == "__main__":
    download_datasets()
