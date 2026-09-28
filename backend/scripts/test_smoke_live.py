import requests
import json
import os
import cv2
import numpy as np

API_URL = "http://127.0.0.1:8000/api/v1/diagnose/"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_endpoint(image_path, case_name):
    print(f"\n--- Testing {case_name} ---")
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    import jwt
    token = jwt.encode({"sub": "test_integration_user"}, "your-super-secret-jwt-token-with-at-least-32-characters-long", algorithm="HS256")
    headers = {'Authorization': f'Bearer {token}'}
    
    with open(image_path, 'rb') as f:
        files = {'image': ('test.jpg', f, 'image/jpeg')}
        try:
            resp = requests.post(API_URL, files=files, headers=headers)
            print(f"Status Code: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Prediction: {data.get('predicted_disease', 'N/A')}")
                print(f"Confidence: {data.get('confidence', 'N/A')}")
                print(f"Low Confidence Flag: {data.get('is_low_confidence', 'N/A')}")
                print(f"Action Priority: {data.get('action_priority', 'N/A')}")
                print(f"Has Grad-CAM: {'gradcam_image_base64' in data}")
                print(f"Has Action Plan: {'action_plan' in data}")
            else:
                print(resp.text)
        except Exception as e:
            print(f"Request failed: {e}")

def main():
    # Find some test images
    import glob
    
    # 1. Healthy Case
    healthy_img = os.path.join(REPO_ROOT, "backend/data/external_field/chilli_india/Chili Leaf Disease Original Dataset/Healthy/Healthy00001.JPG")
    diseased_img = os.path.join(REPO_ROOT, "backend/data/external_field/chilli_india/Chili Leaf Disease Original Dataset/Bacterial Spot/Bacterial Spot00001.JPG")
    
    # Create temp images for cases 3 and 4
    temp_dir = os.path.join(REPO_ROOT, "backend/data/scratch")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 3. Low-confidence (Gaussian noise added to healthy)
    if os.path.exists(healthy_img):
        img = cv2.imread(healthy_img)
        noise = np.random.normal(0, 50, img.shape).astype(np.uint8)
        low_conf = cv2.add(img, noise)
        low_conf_img = os.path.join(temp_dir, "low_conf.jpg")
        cv2.imwrite(low_conf_img, low_conf)
    else:
        low_conf_img = "low_conf.jpg"
        
    # 4. Poor-quality case (Black image / non-plant)
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    poor_quality_img = os.path.join(temp_dir, "poor_quality.jpg")
    cv2.imwrite(poor_quality_img, black_img)
    
    print("LIVE SMOKE TESTS")
    test_endpoint(healthy_img, "Healthy Case")
    test_endpoint(diseased_img, "Diseased Case")
    test_endpoint(low_conf_img, "Low-Confidence Case")
    test_endpoint(poor_quality_img, "Poor-Quality/OOD Case")

if __name__ == '__main__':
    main()
