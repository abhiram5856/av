import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
from sklearn.metrics import accuracy_score

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    model = load_model(model_path, device)
    
    LAB_TEST_PATH = os.path.join(REPO_ROOT, "evaluation", "clean_test_split.json")
    with open(LAB_TEST_PATH) as f:
        lab_data = json.load(f)
    
    paths = []
    if isinstance(lab_data, dict):
        for k, v in lab_data.items():
            paths.extend(v[:30]) # ~900 images
    else:
        paths = [x['path'] for x in lab_data][:900]
        
    confs = []
    corrects = []
    
    print("Running Calibration Test...")
    with torch.inference_mode():
        for path in paths:
            cls_name = os.path.basename(os.path.dirname(path))
            if cls_name not in CLASS_TO_IDX: continue
            label = CLASS_TO_IDX[cls_name]
            
            p_lab = os.path.join(REPO_ROOT, path)
            if not os.path.exists(p_lab): continue
            
            try:
                img = Image.open(p_lab).convert("RGB")
            except: continue
            
            tta_outs = []
            for t in tta_transforms:
                tensor = t(img).unsqueeze(0).to(device)
                tta_outs.append(F.softmax(model(tensor), dim=1))
            
            avg_probs = torch.stack(tta_outs).mean(dim=0)[0]
            pred = torch.argmax(avg_probs).item()
            conf = float(avg_probs[pred].item())
            
            confs.append(conf)
            corrects.append(1 if pred == label else 0)

    confs = np.array(confs)
    corrects = np.array(corrects)
    
    # Calculate ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    
    report = "# Uncertainty Calibration Report\n\n"
    report += "| Bin | Accuracy | Avg Confidence | Samples |\n"
    report += "|---|---|---|---|\n"
    
    for i in range(10):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if i == 9: mask = (confs >= bins[i]) & (confs <= bins[i+1])
        
        n = np.sum(mask)
        if n > 0:
            acc = np.mean(corrects[mask])
            conf_avg = np.mean(confs[mask])
            ece += (n / len(confs)) * np.abs(acc - conf_avg)
            report += f"| {bins[i]:.1f}-{bins[i+1]:.1f} | {acc:.2%} | {conf_avg:.2%} | {n} |\n"
    
    coverage_80 = np.mean(confs >= 0.80)
    selective_acc_80 = np.mean(corrects[confs >= 0.80]) if np.sum(confs >= 0.80) > 0 else 0
    coverage_60 = np.mean(confs >= 0.60)
    selective_acc_60 = np.mean(corrects[confs >= 0.60]) if np.sum(confs >= 0.60) > 0 else 0
    
    report += f"\n## Overall Metrics\n"
    report += f"- **Expected Calibration Error (ECE):** {ece:.4f}\n"
    report += f"- **Threshold 0.80:** Coverage = {coverage_80:.2%}, Selective Accuracy = {selective_acc_80:.2%}\n"
    report += f"- **Threshold 0.60:** Coverage = {coverage_60:.2%}, Selective Accuracy = {selective_acc_60:.2%}\n"
    
    out_path = os.path.join(REPO_ROOT, "evaluation", "uncertainty_calibration_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print("Saved calibration report to", out_path)

if __name__ == '__main__':
    main()
