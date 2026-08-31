"""
End-to-End API Verification
===========================
Tests the FastAPI /api/v1/diagnose endpoint using a dummy image payload
to ensure the TRACE-RCE v3 integration executes successfully.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
sys.path.append(str(REPO_ROOT))

# Import the FastAPI app
from backend.main import app
from backend.auth.security import verify_token

# Override authentication for testing
app.dependency_overrides[verify_token] = lambda: "test_user_id"

client = TestClient(app)

def test_diagnose_endpoint():
    print("Testing POST /api/v1/diagnose...")
    
    # Create a dummy image file payload
    # Just sending a 1x1 pixel image or empty file as bytes
    dummy_image_bytes = b"dummy_image_content_for_testing"
    
    files = {
        "image": ("test_leaf.jpg", dummy_image_bytes, "image/jpeg")
    }
    
    # Optional metadata as expected by upload.py / diagnose.py
    data = {
        "latitude": 17.3850, # Hyderabad, Telangana
        "longitude": 78.4867
    }
    
    response = client.post("/api/v1/diagnose", files=files, data=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n[SUCCESS] Response parsed successfully!")
        
        # We expect a nested 'root_cause' object populated by TRACERootCauseEngineV3
        rc_result = result.get("root_cause", {})
        print("\n--- Root Cause Result Schema ---")
        print(f"Primary Cause: {rc_result.get('primary_cause')}")
        print(f"Confidence: {rc_result.get('confidence_score')}")
        print(f"Reasoning: {rc_result.get('reasoning_chain')}")
        print(f"Contributing Factors: {rc_result.get('contributing_factors')}")
        
    else:
        print("\n[FAILED] Endpoint returned an error:")
        print(response.text)

if __name__ == "__main__":
    test_diagnose_endpoint()
