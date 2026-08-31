import os
import json
import torch
import numpy as np
import cv2
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

import sys
sys.path.append(os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI"))

from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG
from backend.models.explainability import GradCAM

EVAL_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\evaluation")
TEST_SPLIT_PATH = os.path.join(EVAL_DIR, "test_split.json")
WEIGHTS_PATH = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights\best_baseline.pth")
DOCS_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\docs")
IMG_OUT_DIR = os.path.join(DOCS_DIR, "images", "gradcam")

def build_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    return model

def run_gradcam_audit():
    os.makedirs(IMG_OUT_DIR, exist_ok=True)
    
    if not os.path.exists(TEST_SPLIT_PATH):
        print(f"No test split at {TEST_SPLIT_PATH}")
        return
        
    with open(TEST_SPLIT_PATH, "r") as f:
        records = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    cam = GradCAM(model, model.features[-1])

    mean, std = MODEL_CONFIG["normalize_mean"], MODEL_CONFIG["normalize_std"]
    size = MODEL_CONFIG["input_size"][0]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    categories = {
        "correct_high": [],
        "correct_low": [],
        "incorrect_high": [],
        "incorrect_low": []
    }

    print("Scanning test set to find representative samples...")
    # Scan records to find matching samples
    for record in records:
        if all(len(v) >= 4 for v in categories.values()):
            break # Got enough samples
            
        img_path = record["path"]
        true_label = record["class_idx"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue
            
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, pred_label = torch.max(probs, 0)
            
            conf = float(conf)
            pred_label = int(pred_label)
            
        is_correct = (pred_label == true_label)
        
        entry = {
            "path": img_path,
            "true_name": CLASS_NAMES[true_label],
            "pred_name": CLASS_NAMES[pred_label],
            "conf": conf,
            "image": image,
            "tensor": input_tensor
        }
        
        if is_correct and conf >= 0.8 and len(categories["correct_high"]) < 4:
            categories["correct_high"].append(entry)
        elif is_correct and conf < 0.5 and len(categories["correct_low"]) < 4:
            categories["correct_low"].append(entry)
        elif not is_correct and conf >= 0.8 and len(categories["incorrect_high"]) < 4:
            categories["incorrect_high"].append(entry)
        elif not is_correct and conf < 0.5 and len(categories["incorrect_low"]) < 4:
            categories["incorrect_low"].append(entry)

    print("Generating heatmaps...")
    
    md_content = "# GRADCAM BASELINE AUDIT\n\n"
    md_content += "This document verifies whether the baseline MobileNetV3-Small attends to actual disease lesions (causally relevant regions) or relies on background artifacts.\n\n"

    for cat_name, entries in categories.items():
        md_content += f"## Category: {cat_name.replace('_', ' ').title()}\n\n"
        if not entries:
            md_content += "*No samples found for this category.*\n\n"
            continue
            
        for i, entry in enumerate(entries):
            # Generate CAM
            heatmap, _, _ = cam.generate_heatmap(entry["tensor"], class_idx=None)
            
            # Overlay
            cv_img = cv2.cvtColor(np.array(entry["image"]), cv2.COLOR_RGB2BGR)
            heatmap_resized = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed_img = heatmap_color * 0.4 + cv_img * 0.6
            
            out_name = f"{cat_name}_{i}.jpg"
            out_path = os.path.join(IMG_OUT_DIR, out_name)
            cv2.imwrite(out_path, superimposed_img)
            
            md_content += f"### Sample {i+1}\n"
            md_content += f"- **True Class:** {entry['true_name']}\n"
            md_content += f"- **Predicted Class:** {entry['pred_name']}\n"
            md_content += f"- **Confidence:** {entry['conf']:.4f}\n"
            md_content += f"![GradCAM](images/gradcam/{out_name})\n\n"

    md_path = os.path.join(DOCS_DIR, "GRADCAM_BASELINE_AUDIT.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"GradCAM audit generated at {md_path}")

if __name__ == "__main__":
    run_gradcam_audit()
