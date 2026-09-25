import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import ImageFilter, ImageEnhance
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from backend.models.class_registry import NUM_CLASSES, CLASS_TO_IDX, MODEL_CONFIG
from backend.api.diagnose import tta_transforms

def load_model(path, device):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def apply_perturbation(img, p_type):
    if p_type == "original":
        return img
    elif p_type == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif p_type == "dark":
        return ImageEnhance.Brightness(img).enhance(0.4)
    elif p_type == "bright":
        return ImageEnhance.Brightness(img).enhance(1.8)
    elif p_type == "low_contrast":
        return ImageEnhance.Contrast(img).enhance(0.3)
    elif p_type == "jpeg_compress":
        import io
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=15)
        return Image.open(b)
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    model = load_model(model_path, device)
    
    # We will test on a small subset of the lab test set to save time.
    LAB_TEST_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_test_split.json")
    with open(LAB_TEST_PATH) as f:
        lab_data = json.load(f)
    
    paths = []
    if isinstance(lab_data, dict):
        for k, v in lab_data.items():
            paths.extend(v[:10]) # Limit to 10 per class to keep it fast
    else:
        paths = [x['path'] for x in lab_data][:300]
        
    perturbations = ["original", "blur", "dark", "bright", "low_contrast", "jpeg_compress"]
    results = {p: {"correct": 0, "total": 0, "confs": [], "high_conf_errors": 0} for p in perturbations}
    
    print("Running Robustness Stress Test on", len(paths), "images.")
    
    with torch.inference_mode():
        for path in paths:
            cls_name = os.path.basename(os.path.dirname(path))
            if cls_name not in CLASS_TO_IDX: continue
            label = CLASS_TO_IDX[cls_name]
            
            p_lab = os.path.join(REPO_ROOT, path)
            if not os.path.exists(p_lab): continue
            
            try:
                base_img = Image.open(p_lab).convert("RGB")
            except: continue
            
            for p_type in perturbations:
                img = apply_perturbation(base_img.copy(), p_type)
                
                # Use TTA
                tta_outs = []
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0).to(device)
                    tta_outs.append(F.softmax(model(tensor), dim=1))
                
                avg_probs = torch.stack(tta_outs).mean(dim=0)[0]
                pred = torch.argmax(avg_probs).item()
                conf = float(avg_probs[pred].item())
                
                results[p_type]["total"] += 1
                results[p_type]["confs"].append(conf)
                if pred == label:
                    results[p_type]["correct"] += 1
                else:
                    if conf > 0.8:
                        results[p_type]["high_conf_errors"] += 1

    report = "# Real-World Robustness Report\n\n"
    report += "This suite evaluates the model's resilience to common smartphone camera artifacts.\n\n"
    report += "| Perturbation | Accuracy | Avg Confidence | High Conf Errors (>=0.8) |\n"
    report += "|---|---|---|---|\n"
    
    for p in perturbations:
        data = results[p]
        if data["total"] == 0: continue
        acc = data["correct"] / data["total"]
        avg_c = np.mean(data["confs"])
        hce = data["high_conf_errors"]
        report += f"| {p.capitalize()} | {acc:.2%} | {avg_c:.2%} | {hce} |\n"
        
    out_path = os.path.join(REPO_ROOT, "evaluation", "real_world_robustness_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Saved robustness report to", out_path)

if __name__ == '__main__':
    main()
