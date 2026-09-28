import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from collections import defaultdict
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from backend.models.class_registry import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES
from backend.api.diagnose import tta_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def phase1_coverage_analysis():
    print("PHASE 1: Data Coverage Analysis")
    coverage = {cls: {'lab_train': 0, 'field_train': 0, 'field_test': 0, 'external_original': 0, 'external_derived': 0} for cls in CLASS_NAMES}
    
    # 1. Field Splits (Internal)
    field_splits_path = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    if os.path.exists(field_splits_path):
        with open(field_splits_path) as f:
            fs = json.load(f)
            for p in fs.get('trainval', []):
                cls = os.path.basename(os.path.dirname(p))
                if cls in coverage: coverage[cls]['field_train'] += 1
            for p in fs.get('test', []):
                cls = os.path.basename(os.path.dirname(p))
                if cls in coverage: coverage[cls]['field_test'] += 1

    # 2. External Original
    ext_orig_path = os.path.join(REPO_ROOT, "evaluation", "external_original_field_candidates.json")
    if os.path.exists(ext_orig_path):
        with open(ext_orig_path) as f:
            eo = json.load(f)
            for item in eo:
                cls = item.get('canonical_class')
                if cls in coverage: coverage[cls]['external_original'] += 1
                
    # 3. External Derived
    ext_deriv_path = os.path.join(REPO_ROOT, "evaluation", "external_derived_field_candidates.json")
    if os.path.exists(ext_deriv_path):
        with open(ext_deriv_path) as f:
            ed = json.load(f)
            for item in ed:
                cls = item.get('canonical_class')
                if cls in coverage: coverage[cls]['external_derived'] += 1
                
    # Lab train estimation (could scan directories, but just to get a baseline we can assume PlantVillage averages)
    pv_dir = os.path.join(REPO_ROOT, "backend", "data", "processed_dataset")
    if os.path.exists(pv_dir):
        for cls in CLASS_NAMES:
            d = os.path.join(pv_dir, cls)
            if os.path.isdir(d):
                coverage[cls]['lab_train'] = len([f for f in os.listdir(d) if f.lower().endswith('.jpg')])
                
    return coverage

class FieldTestDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.samples = []
        for p in paths:
            cls_name = os.path.basename(os.path.dirname(p))
            if cls_name in CLASS_TO_IDX:
                if not os.path.isabs(p):
                    p_field = os.path.join(REPO_ROOT, "backend", "data", "processed_field_dataset", p)
                    if os.path.exists(p_field): p = p_field
                if os.path.exists(p):
                    self.samples.append((p, CLASS_TO_IDX[cls_name]))
        self.transform = transform
        
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label, path

def phase2_error_analysis():
    print("PHASE 2: Error Analysis")
    
    field_splits_path = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    with open(field_splits_path) as f:
        field_test_list = json.load(f)['test']
        
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ds = FieldTestDataset(field_test_list, transform)
    loader = DataLoader(ds, batch_size=1)
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(REPO_ROOT, "backend", "models", "weights", "nova_mobilenet_v3_34_classes.pth"), map_location=device))
    model.to(device)
    model.eval()
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for views, target, _ in loader: # TTA not fully applied here for speed, just using center crop for confusion matrix to find weak spots
            # for full TTA it's slower. Let's do TTA since it's the authoritative behavior.
            path = _[0]
            with Image.open(path) as img:
                img = img.convert('RGB')
                tta_outs = []
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0).to(device)
                    tta_outs.append(torch.nn.functional.softmax(model(tensor), dim=1))
            avg = torch.stack(tta_outs).mean(dim=0)
            pred = torch.argmax(avg, dim=1).item()
            all_targets.append(target.item())
            all_preds.append(pred)
            
    cr = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return cr, cm

def main():
    coverage = phase1_coverage_analysis()
    cr, cm = phase2_error_analysis()
    
    with open(os.path.join(REPO_ROOT, "evaluation", "phase2_data_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=4)
        
    with open(os.path.join(REPO_ROOT, "evaluation", "phase2_error_metrics.json"), "w") as f:
        json.dump(cr, f, indent=4)
        
    # Write MD report
    report_path = os.path.join(REPO_ROOT, "evaluation", "phase2_field_error_analysis.md")
    with open(report_path, "w") as f:
        f.write("# Phase 2: Field Error Analysis\n\n")
        f.write("## Data Coverage Matrix\n")
        f.write("| Class | Lab Train | Field Train | Ext Orig | Ext Deriv | Field Test |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cls in CLASS_NAMES:
            d = coverage[cls]
            f.write(f"| {cls} | {d['lab_train']} | {d['field_train']} | {d['external_original']} | {d['external_derived']} | {d['field_test']} |\n")
            
        f.write("\n## Weakest Field Classes\n")
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        
        weak_classes = []
        for cls in CLASS_NAMES:
            metrics = cr.get(cls, {})
            if metrics.get('f1-score', 0) < 0.70 or metrics.get('support', 0) == 0:
                weak_classes.append((cls, metrics))
                
        weak_classes.sort(key=lambda x: x[1].get('f1-score', 0))
        for cls, metrics in weak_classes:
            f.write(f"| {cls} | {metrics.get('precision', 0):.2f} | {metrics.get('recall', 0):.2f} | {metrics.get('f1-score', 0):.2f} | {metrics.get('support', 0)} |\n")
            
        f.write("\n## Major Confusions\n")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if i != j and cm[i][j] > 0:
                    f.write(f"- {CLASS_NAMES[i]} misclassified as {CLASS_NAMES[j]}: {cm[i][j]} times\n")

if __name__ == '__main__':
    main()
