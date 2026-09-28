import requests
import json
import jwt
from datetime import datetime, timedelta

def create_test_token():
    SECRET = "super_secret_jwt_key_for_agrivision_123!"
    payload = {
        "sub": "testuser",
        "role": "farmer",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

token = create_test_token()
print("Token:", token)

# Ensure the server is running on port 8000
res = requests.post(
    "http://127.0.0.1:8000/api/v1/diagnose/",
    files={"image": ("test_disease.jpg", open("test_disease.jpg", "rb"), "image/jpeg")},
    headers={"Authorization": f"Bearer {token}"}
)

if res.status_code == 200:
    print(json.dumps(res.json(), indent=2))
else:
    print("Error", res.status_code, res.text)
