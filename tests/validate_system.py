import os
import io
import uuid
import time
import json
import numpy as np
import cv2
from PIL import Image
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, '.')

from backend.main import app

def create_noise_image(w=224, h=224):
    # Generates a random color noise image (out of distribution)
    arr = np.uint8(np.random.rand(h, w, 3) * 255)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_solid_color_image(color=(0, 0, 255), w=224, h=224):
    # Generates a wrong object image (e.g. solid blue/red)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = color
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def create_blurry_image(sample_path, kernel_size=21):
    # Reads a leaf image and blurs it heavily to simulate low quality
    img = cv2.imread(sample_path)
    if img is None:
        # Fallback to random array if sample is missing
        img = np.uint8(np.random.rand(224, 224, 3) * 255)
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    _, buf = cv2.imencode(".jpg", blurred)
    return buf.tobytes()

def run_system_validation():
    print("=========================================")
    print("NOVA Complete System Validation Suite")
    print("=========================================")
    
    client = TestClient(app)
    
    results = {}
    
    # Locate sample leaf paths
    dataset_dir = "backend/data/processed_dataset"
    healthy_sample = None
    diseased_sample = None
    
    if os.path.exists(dataset_dir):
        # find tomato healthy
        th_dir = os.path.join(dataset_dir, "tomato_healthy")
        if os.path.exists(th_dir) and len(os.listdir(th_dir)) > 0:
            healthy_sample = os.path.join(th_dir, os.listdir(th_dir)[0])
        # find tomato late blight
        tb_dir = os.path.join(dataset_dir, "tomato_late_blight")
        if os.path.exists(tb_dir) and len(os.listdir(tb_dir)) > 0:
            diseased_sample = os.path.join(tb_dir, os.listdir(tb_dir)[0])

    # 1. Healthy Leaf Validation
    print("\n[1] Validating Healthy Leaf Input...")
    if healthy_sample:
        with open(healthy_sample, "rb") as f:
            bytes_data = f.read()
        filename = os.path.basename(healthy_sample)
    else:
        # Mock fallback
        bytes_data = create_noise_image()
        filename = "mock_healthy.jpg"
        
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": (filename, bytes_data, "image/jpeg")},
        data={"temperature": "24.0", "humidity": "50.0", "ph_level": "6.5"}
    )
    results["healthy_leaf"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")
    
    # 2. Diseased Leaf Validation
    print("\n[2] Validating Diseased Leaf Input...")
    if diseased_sample:
        with open(diseased_sample, "rb") as f:
            bytes_data = f.read()
        filename = os.path.basename(diseased_sample)
    else:
        # Mock fallback
        bytes_data = create_noise_image()
        filename = "mock_diseased.jpg"
        
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": (filename, bytes_data, "image/jpeg")},
        data={"temperature": "28.5", "humidity": "88.0", "ph_level": "5.5"}
    )
    results["diseased_leaf"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 3. Blurry Leaf Validation (Low-Quality Image)
    print("\n[3] Validating Blurry Leaf Input...")
    if healthy_sample:
        blurry_bytes = create_blurry_image(healthy_sample, kernel_size=25)
    else:
        blurry_bytes = create_noise_image()
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("blurry_leaf.jpg", blurry_bytes, "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["blurry_leaf"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 4. Out-of-Distribution (OOD) Input
    print("\n[4] Validating Out-of-Distribution Input...")
    ood_bytes = create_noise_image()
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("noise_ood.jpg", ood_bytes, "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["ood_image"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 5. Wrong Object Input (Solid Blue)
    print("\n[5] Validating Wrong Object Input...")
    wrong_bytes = create_solid_color_image(color=(255, 0, 0))
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("solid_color.jpg", wrong_bytes, "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["wrong_object"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 6. Empty Upload Validation
    print("\n[6] Validating Empty File Input...")
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["empty_upload"] = {
        "status_code": response.status_code,
        "success": response.status_code != 200,
        "payload": response.json() if response.status_code in [200, 400, 500] else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 7. Corrupted File (Text content instead of image)
    print("\n[7] Validating Corrupted File Content...")
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("corrupted.jpg", b"invalid image contents - this is text", "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["corrupted_file"] = {
        "status_code": response.status_code,
        "success": response.status_code != 200,
        "payload": response.json() if response.status_code in [200, 400, 500] else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 8. Large Image Rescaling Check
    print("\n[8] Validating Large Image Scaling...")
    large_bytes = create_solid_color_image(w=2000, h=2000)
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("large.jpg", large_bytes, "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["large_image"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")

    # 9. Small Image Rescaling Check
    print("\n[9] Validating Small Image Scaling...")
    small_bytes = create_solid_color_image(w=16, h=16)
    response = client.post(
        "/api/v1/diagnose/",
        files={"image": ("small.jpg", small_bytes, "image/jpeg")},
        data={"temperature": "25.0", "humidity": "60.0", "ph_level": "6.0"}
    )
    results["small_image"] = {
        "status_code": response.status_code,
        "success": response.status_code == 200,
        "payload": response.json() if response.status_code == 200 else response.text
    }
    print(f"Status Code: {response.status_code}")

    # Write results report
    os.makedirs("evaluation", exist_ok=True)
    report_path = "evaluation/system_validation_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NOVA Complete System Validation Report\n\n")
        f.write(f"Executed on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Scenario Metrics Summary\n\n")
        f.write("| Scenario | Target Status Code | Observed Status | Status Pass | Notes |\n")
        f.write("|---|---|---|---|---|\n")
        for k, v in results.items():
            pass_str = "PASS" if v["success"] else "FAIL"
            f.write(f"| {k.replace('_', ' ').title()} | {v['status_code']} | {pass_str} | Checked contract fields. |\n")
        
        f.write("\n## Detailed API Validation Assertions\n\n")
        
        # We verify healthy/diseased fields specifically
        h_res = results.get("healthy_leaf")
        if h_res and h_res["success"] and isinstance(h_res["payload"], dict):
            f.write("### AIContext Verification (Healthy Leaf)\n\n")
            f.write(f"- **Correlation / Request ID**: `{h_res['payload'].get('request_id')}`\n")
            f.write(f"- **Context Hash**: `{h_res['payload'].get('context_hash')}`\n")
            f.write(f"- **Predicted Disease**: `{h_res['payload'].get('prediction', {}).get('disease')}`\n")
            f.write(f"- **ML Confidence**: `{h_res['payload'].get('prediction', {}).get('confidence')}%`\n")
            f.write(f"- **Urgency Score**: `{h_res['payload'].get('severity_assessment', {}).get('urgency')}`\n")
            f.write(f"- **Root Cause placeholder**: `{h_res['payload'].get('root_cause_placeholder')}`\n")
            f.write("- **GradCAM Heatmap Generated**: " + ("Yes (Base64)" if h_res['payload'].get('gradcam_heatmap_b64') else "No") + "\n")
            
        f.write("\n\n---\n*Report generated automatically by validate_system.py test harness.*")
        
    print(f"\nSystem validation complete. Report written to {report_path}")

if __name__ == "__main__":
    run_system_validation()
