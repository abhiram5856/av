import os
import sys
import json
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.main import app

client = TestClient(app)

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
            "user_id": "test_integration_user"
        }
        
        print("Sending request to /api/v1/diagnose/")
        response = client.post("/api/v1/diagnose/", files=files, data=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            res_json = response.json()
            print("Response Keys:", res_json.keys())
            print("\nPrediction:")
            print(json.dumps(res_json.get("prediction"), indent=2))
            
            print("\nSeverity:")
            print(json.dumps(res_json.get("severity"), indent=2))
            
            print("\nRoot Cause Analysis (Top Cause):")
            rca = res_json.get("root_cause_analysis", {})
            if rca and "ranked_causes" in rca and len(rca["ranked_causes"]) > 0:
                print(json.dumps(rca["ranked_causes"][0], indent=2))
            else:
                print("No ranked causes returned.")
                print(json.dumps(rca, indent=2))
                
            print("\nReasoning Chain:")
            print(json.dumps(rca.get("reasoning_chain"), indent=2))
        else:
            print("Error details:", response.text)

if __name__ == "__main__":
    test_diagnose_endpoint()
