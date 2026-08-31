import os
import urllib.request
import zipfile
import re
import shutil

url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
zip_path = "plantdoc.zip"
extract_dir = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\raw_field_data\plantdoc_extracted")

def sanitize_filename(filename):
    # Remove invalid Windows characters
    return re.sub(r'[<>:"/\|?*]', '_', filename)

def download_and_extract():
    print("Downloading PlantDoc...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting and sanitizing filenames...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip directories
            if file_info.is_dir():
                continue
                
            # Sanitize the path
            parts = file_info.filename.split('/')
            sanitized_parts = [sanitize_filename(p) for p in parts]
            sanitized_path = os.path.join(extract_dir, *sanitized_parts)
            
            # Create subdirectories
            os.makedirs(os.path.dirname(sanitized_path), exist_ok=True)
            
            # Extract file
            with zip_ref.open(file_info) as source, open(sanitized_path, "wb") as target:
                shutil.copyfileobj(source, target)
                
    print(f"Successfully extracted PlantDoc to {extract_dir}")
    if os.path.exists(zip_path):
        os.remove(zip_path)

if __name__ == "__main__":
    download_and_extract()
