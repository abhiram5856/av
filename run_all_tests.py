import os
import sys
import json
from fastapi.testclient import TestClient
from jose import jwt
sys.path.append(r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI')
from backend.main import app

client = TestClient(app)

def get_test_token():
    return jwt.encode({'sub': 'test_integration_user'}, 'your-super-secret-jwt-token-with-at-least-32-characters-long', algorithm='HS256')

images = {
    'Diseased (Lab)': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_late_blight\plantvillagedataset_0.JPG',
    'Healthy (Lab)': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_healthy\plantvillagedataset_0.JPG',
    'Field': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_field_dataset\rice_leaf_blast\DSC_0101.JPG'
}

data = {
    'temperature': 26.5,
    'humidity': 85.0,
    'ph_level': 6.2,
    'latitude': 17.3850,
    'longitude': 78.4867,
    'user_id': 'test_integration_user'
}
headers = {'Authorization': f'Bearer {get_test_token()}'}

for name, path in images.items():
    print(f'\n--- Testing: {name} ---')
    if not os.path.exists(path):
        print(f'File not found: {path}')
        continue
    
    with open(path, 'rb') as f:
        files = {'image': ('leaf.jpg', f, 'image/jpeg')}
        response = client.post('/api/v1/diagnose/', files=files, data=data, headers=headers)
        
        print(f'Status Code: {response.status_code}')
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Class: {result.get('disease')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Has Grad-CAM: {bool(result.get('grad_cam'))}")
            print(f"Concern Score: {result.get('concern_score')}")
            rag_context = result.get('rag_context', {})
            print(f"Has RAG Treatment: {bool(rag_context.get('treatment'))}")
        else:
            print(response.text)

print('\nBackend tests complete.')
