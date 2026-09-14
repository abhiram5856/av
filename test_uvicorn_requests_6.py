import requests
from jose import jwt

def get_test_token():
    return jwt.encode({'sub': 'test_integration_user'}, 'your-super-secret-jwt-token-with-at-least-32-characters-long', algorithm='HS256')

images = {
    'Rice Field': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_field_dataset\rice_leaf_blast\paddydoctor_100004.jpg'
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
    try:
        with open(path, 'rb') as f:
            files = {'image': ('leaf.jpg', f, 'image/jpeg')}
            response = requests.post('http://localhost:8003/api/v1/diagnose/', files=files, data=data, headers=headers)
            
            print(f'Status Code: {response.status_code}')
            if response.status_code == 200:
                result = response.json()
                ai_context = result.get('ai_context', {})
                print(f"Predicted Class: {ai_context.get('vision', {}).get('predicted_disease')}")
                print(f"Concern Score: {ai_context.get('concern', {}).get('concern_score')}")
                print(f"RAG Recommendation: {ai_context.get('knowledge', {}).get('context_text_block')}")
            else:
                print(response.text)
    except Exception as e:
        print(f'Request failed: {e}')
