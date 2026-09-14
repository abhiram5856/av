import requests
from jose import jwt
import json
import time

def get_test_token():
    return jwt.encode({"sub": "test_integration_user"}, "your-super-secret-jwt-token-with-at-least-32-characters-long", algorithm="HS256")

headers = {"Authorization": f"Bearer {get_test_token()}"}

test_images = [
    {
        "desc": "Diseased Lab Image (Tomato Late Blight)",
        "path": r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_late_blight\plantvillagedataset_0.JPG"
    },
    {
        "desc": "Healthy Image (Tomato Healthy)",
        "path": r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_healthy\plantvillagedataset_0.JPG"
    },
    {
        "desc": "Field Style / Lower Quality Image (Rice Leaf Blast)",
        "path": r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\rice_leaf_blast\ricedisease_0.jpg"
    }
]

print("Starting Validation Test...\n")
for idx, test in enumerate(test_images, 1):
    print(f"--- Test {idx}: {test['desc']} ---")
    try:
        with open(test['path'], "rb") as f:
            files = {"image": ("leaf.jpg", f, "image/jpeg")}
            data = {
                "temperature": 28.5,
                "humidity": 75.0,
                "ph_level": 6.5,
                "latitude": 17.3850,
                "longitude": 78.4867,
                "growth_stage": "Flowering"
            }
            
            start_time = time.time()
            response = requests.post("http://localhost:8000/api/v1/diagnose/", files=files, data=data, headers=headers)
            end_time = time.time()
            
            print(f"Status: {response.status_code} ({end_time-start_time:.2f}s)")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Diagnosis ID: {result.get('diagnosis_id')}")
                
                # Check top components
                pred = result.get('prediction', {})
                print(f"Prediction: {pred.get('disease')} (Conf: {pred.get('confidence')}%)")
                
                # Grad-CAM
                print(f"Has Grad-CAM: {bool(result.get('grad_cam'))}")
                
                # Concern Score
                concern = result.get('concern', {})
                print(f"Concern Score: {concern.get('score')} ({concern.get('level')})")
                
                # Recommendation (RAG)
                rec = result.get('recommendation', {})
                sources = rec.get('retrieved_sources', [])
                rag_resp = rec.get('rag_response', {})
                print(f"RAG Used: {len(sources)} sources retrieved.")
                print(f"Treatment snippet: {str(rag_resp.get('treatment', ''))[:100]}...")
                
                # Flags
                print(f"Flags: {result.get('flags', [])}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")
    print("\n")
