import requests
import json
import jwt
import glob
from datetime import datetime, timedelta

SECRET = 'your-super-secret-jwt-token-with-at-least-32-characters-long'
token = jwt.encode({'sub': 'testuser', 'exp': datetime.utcnow() + timedelta(hours=1)}, SECRET, algorithm='HS256')
headers = {'Authorization': f'Bearer {token}'}

cases = [
    ('tomato_early_blight', glob.glob('backend/data/processed_dataset/tomato_early_blight/*.jpg')),
    ('rice_leaf_blast', glob.glob('backend/data/processed_dataset/rice_leaf_blast/*.jpg')),
    ('potato_late_blight', glob.glob('backend/data/processed_dataset/potato_late_blight/*.jpg')),
    ('tomato_healthy', glob.glob('backend/data/processed_dataset/tomato_healthy/*.jpg')),
    ('rice_bacterial_leaf_blight', glob.glob('backend/data/processed_dataset/rice_bacterial_leaf_blight/*.jpg')),
    ('cotton_leaf_curl_virus', glob.glob('backend/data/processed_dataset/cotton_leaf_curl_virus/*.jpg')),
]

results = []
for class_name, images in cases:
    if not images:
        print(f'SKIP {class_name}: no images found')
        continue
    img_path = images[0]
    res = requests.post(
        'http://127.0.0.1:8000/api/v1/diagnose/',
        files={'image': ('leaf.jpg', open(img_path, 'rb'), 'image/jpeg')},
        headers=headers
    )
    if res.status_code == 200:
        data = res.json()
        v = data.get('VISION', {})
        k = data.get('KNOWLEDGE', {})
        d = data.get('DECISION', {})
        summary = k.get('disease_specific_knowledge', {})
        print(f'--- {class_name} ---')
        print(f'  prediction: {v.get("prediction")} ({v.get("confidence")}%)')
        print(f'  disease_type: {summary.get("disease_type")}')
        print(f'  distinctive_pattern: {str(summary.get("distinctive_pattern", ""))[:80]}')
        print(f'  differentials: {v.get("differential_conditions", [])}')
        print(f'  action_priority: {d.get("action_priority")}')
        print(f'  knowledge_completeness: {k.get("knowledge_completeness")}')
        print(f'  source_count: {len(k.get("source_references", []))}')
        print(f'  prevention_snippet: {str(k.get("prevention", ""))[:80]}')
        print()
        results.append({'class': class_name, 'status': 'PASS'})
    else:
        print(f'ERROR {class_name}: {res.status_code}')
        results.append({'class': class_name, 'status': 'FAIL'})

print(f'TOTAL: {len(results)} tested')
pass_count = sum(1 for r in results if r['status'] == 'PASS')
print(f'PASS: {pass_count}, FAIL: {len(results) - pass_count}')
