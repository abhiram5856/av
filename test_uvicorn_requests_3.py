import requests
from jose import jwt

def get_test_token():
    return jwt.encode({'sub': 'test_integration_user'}, 'your-super-secret-jwt-token-with-at-least-32-characters-long', algorithm='HS256')

path = r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_late_blight\plantvillagedataset_0.JPG'

data = {
    'temperature': 26.5,
    'humidity': 85.0,
    'ph_level': 6.2,
    'latitude': 17.3850,
    'longitude': 78.4867,
    'user_id': 'test_integration_user'
}
headers = {'Authorization': f'Bearer {get_test_token()}'}

with open(path, 'rb') as f:
    files = {'image': ('leaf.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:8002/api/v1/diagnose/', files=files, data=data, headers=headers)
    print(response.json())
