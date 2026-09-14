import os
import torch
import numpy as np
import time
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
import sys
sys.path.append(str(REPO_ROOT))
from backend.models.class_registry import NUM_CLASSES, CLASS_NAMES, CLASS_TO_IDX, MODEL_CONFIG

def get_size(path):
    return os.path.getsize(path) / (1024 * 1024)

class SimpleDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path) as f:
            data = json.load(f)
        
        self.samples = []
        if isinstance(data, list):
            for item in data:
                cls_name = item["class_name"]
                if cls_name in CLASS_TO_IDX:
                    self.samples.append((item["path"], CLASS_TO_IDX[cls_name]))
        elif isinstance(data, dict):
            for cls_name, paths in data.items():
                if cls_name in CLASS_TO_IDX:
                    for p in paths:
                        self.samples.append((p, CLASS_TO_IDX[cls_name]))
        else:
            raise ValueError("Unsupported JSON format")
            
        self.transform = transform
        
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

def evaluate_model(model_path, device, lab_loader, field_loader):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    def run_eval(loader):
        all_preds = []
        all_targets = []
        all_probs = []
        start = time.perf_counter()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        lat = (time.perf_counter() - start) / len(loader.dataset)
        return all_targets, all_preds, all_probs, lat

    lab_t, lab_p, lab_probs, _ = run_eval(lab_loader)
    field_t, field_p, all_probs, lat = run_eval(field_loader)
    
    return {
        "lab_acc": accuracy_score(lab_t, lab_p),
        "lab_f1": f1_score(lab_t, lab_p, average='macro', zero_division=0),
        "field_acc": accuracy_score(field_t, field_p),
        "field_f1": f1_score(field_t, field_p, average='macro', zero_division=0),
        "field_f1_weighted": f1_score(field_t, field_p, average="weighted", zero_division=0),
        "top3_acc": sum([1 if t in np.argsort(p)[-3:] else 0 for t, p in zip(field_t, all_probs)]) / len(field_t) if len(field_t) > 0 else 0,
        "lat": lat,
        "size": get_size(model_path),
        "field_report": classification_report(field_t, field_p, target_names=CLASS_NAMES, labels=list(range(NUM_CLASSES)), output_dict=True, zero_division=0),
        "field_cm": confusion_matrix(field_t, field_p)
    }

def print_top_confusions(cm, n=10):
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i][j] > 0:
                confusions.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i][j]))
    confusions.sort(key=lambda x: x[2], reverse=True)
    print("Top Confusions:")
    for true_c, pred_c, count in confusions[:n]:
        print(f"True: {true_c:35s} -> Pred: {pred_c:35s} ({count})")

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lab_loader = DataLoader(SimpleDataset(REPO_ROOT / "evaluation" / "clean_test_split.json", transform=val_transform), batch_size=64, shuffle=False)
    
    with open(REPO_ROOT / "backend" / "data" / "field_splits.json") as f:
        field_paths = json.load(f)["test"]
    
    field_data = []
    for p in field_paths:
        cls_name = os.path.basename(os.path.dirname(p))
        if cls_name in CLASS_TO_IDX:
            if not os.path.isabs(p): p = os.path.join(str(REPO_ROOT), "backend", "data", "processed_field_dataset", p)
            field_data.append({"path": p, "class_name": cls_name})
            
    # Write temp json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(field_data, f)
        temp_field_path = f.name
        
    field_loader = DataLoader(SimpleDataset(temp_field_path, transform=val_transform), batch_size=64, shuffle=False)
    
    base_path = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"
    cand_path = REPO_ROOT / "backend" / "models" / "weights" / "candidate_v3.pth"
    
    print("Evaluating Baseline...")
    base_res = evaluate_model(base_path, device, lab_loader, field_loader)
    
    print("Evaluating Candidate...")
    cand_res = evaluate_model(cand_path, device, lab_loader, field_loader)
    
    print("\n================== FINAL COMPARISON ==================")
    print(f"{'Metric':<25} {'Production':<12} {'Candidate':<12} {'Delta':<12}")
    
    def print_metric(name, b_val, c_val, is_pct=True, lower_is_better=False):
        if is_pct:
            b_str, c_str = f"{b_val*100:.2f}%", f"{c_val*100:.2f}%"
            delta = (c_val - b_val)*100
            delta_str = f"{delta:+.2f}%"
        else:
            b_str, c_str = f"{b_val:.4f}", f"{c_val:.4f}"
            delta = c_val - b_val
            delta_str = f"{delta:+.4f}"
            
        print(f"{name:<25} {b_str:<12} {c_str:<12} {delta_str:<12}")

    print_metric("Field Accuracy", base_res['field_acc'], cand_res['field_acc'])
    print_metric("Field Macro F1", base_res['field_f1'], cand_res['field_f1'])
    print_metric("Field Weighted F1", base_res["field_f1_weighted"], cand_res["field_f1_weighted"])
    print_metric("Field Top-3", base_res.get("top3_acc", 0), cand_res.get("top3_acc", 0))
    print_metric("Lab Accuracy", base_res['lab_acc'], cand_res['lab_acc'])
    print_metric("Lab Macro F1", base_res['lab_f1'], cand_res['lab_f1'])
    print_metric("Inference Latency", base_res['lat'], cand_res['lat'], is_pct=False)
    print_metric("Model Size (MB)", base_res['size'], cand_res['size'], is_pct=False)
    
    print("\n--- BASELINE CONFUSIONS ---")
    print_top_confusions(base_res['field_cm'], 5)
    
    print("\n--- CANDIDATE CONFUSIONS ---")
    print_top_confusions(cand_res['field_cm'], 5)
    
    print("\n--- PER-CLASS FIELD METRICS (CANDIDATE vs BASELINE F1) ---")
    for cls in CLASS_NAMES:
        b_f1 = base_res['field_report'][cls]['f1-score']
        c_f1 = cand_res['field_report'][cls]['f1-score']
        print(f"{cls:40s} Base: {b_f1*100:5.1f}% -> Cand: {c_f1*100:5.1f}%  Diff: {(c_f1-b_f1)*100:+5.1f}%")
        
if __name__ == '__main__':
    main()
