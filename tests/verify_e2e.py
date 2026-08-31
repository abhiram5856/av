import sys
import os
import asyncio
from fastapi.testclient import TestClient

# Set path so Python can find backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.main import app

def run_e2e_verification():
    print("=========================================")
    print("NOVA E2E Diagnostic Pipeline Verification")
    print("=========================================")
    
    client = TestClient(app)
    
    # Check health check endpoint
    response = client.get("/health")
    print(f"Health Check: {response.status_code} | Payload: {response.json()}")
    assert response.status_code == 200
    
    # Setup test file payload
    import io
    from PIL import Image
    import numpy as np
    
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Call diagnose API endpoint
    print("Sending POST request to /api/v1/diagnose/ ...")
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("test.jpg", img_byte_arr, "image/jpeg")},
        data={
            "temperature": "24.5",
            "humidity": "85.0",
            "ph_level": "6.0",
            "latitude": "17.3850",
            "longitude": "78.4867",
            "user_id": "usr_test_farmer"
        }
    )
    
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print("ERROR RESPONSE:", response.text)
    assert response.status_code == 200
    
    res_data = response.json()
    print(f"Status: {res_data.get('status')}")
    print(f"Context Hash: {res_data.get('context_hash')}")
    print(f"Predicted Disease: {res_data.get('prediction', {}).get('disease')}")
    print(f"Severity Score: {res_data.get('severity_assessment', {}).get('final_severity_score')}%")
    print(f"Root Cause Engine Placeholder status: {res_data.get('root_cause_placeholder', {}).get('status')}")
    
    # Assert validation
    assert res_data.get("status") == "success"
    assert "ai_context" in res_data
    assert res_data["ai_context"]["system"]["schema_version"] == "1.0.0"
    assert res_data["ai_context"]["user"]["user_id"] == "usr_test_farmer"
    
    print("\nE2E pipeline verification successful!")

if __name__ == "__main__":
    run_e2e_verification()
