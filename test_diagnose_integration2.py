import os
import json
import requests
from jose import jwt

def get_test_token():
    return jwt.encode({"sub": "test_integration_user"}, "your-super-secret-jwt-token-with-at-least-32-characters-long", algorithm="HS256")

def test_diagnose_endpoint():
    image_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_late_blight\plantvillagedataset_0.JPG"
    
    with open(image_path, "rb") as f:
        files = {"image": ("leaf.jpg", f, "image/jpeg")}
        data = {
            "temperature": 26.5,
            "humidity": 85.0,
            "ph_level": 6.2,
            "latitude": 17.3850,
            "longitude": 78.4867,
            "growth_stage": "Unknown"
        }
        headers = {"Authorization": f"Bearer {get_test_token()}"}
        
        print("Sending request to http://localhost:8000/api/v1/diagnose/")
        response = requests.post("http://localhost:8000/api/v1/diagnose/", files=files, data=data, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== DIAGNOSIS RESULTS ===")
            print(f"Diagnosis ID: {result.get('diagnosis_id')}")
            print(f"Predicted Class: {result.get('disease')}")
            print(f"Confidence: {result.get('confidence')}")
            
            rag_context = result.get("rag_context", {})
            print(f"Treatment: {str(rag_context.get('treatment'))[:100]}...")
            
            flags = result.get("flags", [])
            print(f"Flags: {flags}")
            
            print(f"Concern Score: {result.get('concern_score')}")
            print(f"Has Grad-CAM: {bool(result.get('grad_cam'))}")
        else:
            print(f"Error details: {response.text}")

if __name__ == '__main__':
    test_diagnose_endpoint()
