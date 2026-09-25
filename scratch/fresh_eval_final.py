import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from backend.api.diagnose import _build_mobilenet_v3_small, tta_transforms
from backend.models.class_registry import NUM_CLASSES, idx_to_class

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = REPO_ROOT / "evaluation"
WEIGHTS_PATH = REPO_ROOT / "backend" / "models" / "weights" / "nova_mobilenet_v3_34_classes.pth"

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['path']).convert('RGB')
        label = int(item['class_idx'])
        return img, label

def evaluate_split(data_list, model, device):
    all_preds = []
    all_probs = []
    all_labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for item in data_list:
            img = Image.open(item['path']).convert('RGB')
            label = int(item['class_idx'])
            
            tta_outputs = []
            for t in tta_transforms:
                input_tensor = t(img).unsqueeze(0).to(device)
                logits = model(input_tensor)
                probs = F.softmax(logits[0], dim=0)
                tta_outputs.append(probs)
                
            avg_probs = torch.stack(tta_outputs).mean(dim=0)
            class_idx = int(torch.argmax(avg_probs).item())
            
            all_preds.append(class_idx)
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.append(label)
            
    total_time = time.time() - start_time
    latency_ms = (total_time / len(data_list)) * 1000
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate Top-3
    top3_correct = 0
    for i in range(len(all_labels)):
        top3_preds = np.argsort(all_probs[i])[-3:]
        if all_labels[i] in top3_preds:
            top3_correct += 1
    top3_acc = top3_correct / len(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    clf_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top3_acc": top3_acc,
        "latency_ms": latency_ms,
        "cm": cm,
        "clf_report": clf_report
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_mobilenet_v3_small(NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    with open(EVAL_DIR / 'clean_test_split.json', 'r') as f:
        lab_test = json.load(f)
        
    with open(REPO_ROOT / 'backend' / 'data' / 'field_splits.json', 'r') as f:
        field_test_raw = json.load(f)['test']
        
    with open(EVAL_DIR / 'dataset_audit_report.json', 'r') as f:
        class_mapping = json.load(f)['class_mapping']
        
    field_test = []
    for rel_path in field_test_raw:
        class_name = rel_path.split('/')[0]
        class_name = class_name.replace('\\', '/')
        if '/' in class_name: class_name = class_name.split('/')[0]
        
        if class_name not in class_mapping:
            continue
            
        full_path = str(REPO_ROOT / 'backend' / 'data' / 'processed_field_dataset' / rel_path)
        field_test.append({'path': full_path, 'class_idx': class_mapping[class_name]})
        
    print("Evaluating Lab Test...")
    lab_metrics = evaluate_split(lab_test, model, device)
    print("Evaluating Field Test...")
    field_metrics = evaluate_split(field_test, model, device)
    
    print("\n[VERIFIED METRICS]")
    print(f"Field Accuracy = {field_metrics['accuracy'] * 100:.2f}%")
    print(f"Field Macro F1 = {field_metrics['macro_f1'] * 100:.2f}%")
    print(f"Field Weighted F1 = {field_metrics['weighted_f1'] * 100:.2f}%")
    print(f"Field Top-3 = {field_metrics['top3_acc'] * 100:.2f}%")
    print(f"Lab Accuracy = {lab_metrics['accuracy'] * 100:.2f}%")
    print(f"Lab Macro F1 = {lab_metrics['macro_f1'] * 100:.2f}%")
    print(f"Lab Weighted F1 = {lab_metrics['weighted_f1'] * 100:.2f}%")
    
    # Calculate top 10 confusion pairs
    classes = [idx_to_class(i) for i in range(NUM_CLASSES)]
    field_cm = field_metrics['cm']
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j:
                count = field_cm[i][j]
                if count > 0:
                    confusions.append((classes[i], classes[j], count))
    confusions = sorted(confusions, key=lambda x: x[2], reverse=True)[:10]

    per_class_list = []
    for idx, c in enumerate(classes):
        m = field_metrics['clf_report'].get(str(idx), {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0})
        per_class_list.append((c, m['support'], m['precision'], m['recall'], m['f1-score']))
    per_class_list = sorted(per_class_list, key=lambda x: x[4])

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 30))
    fig.patch.set_facecolor('#1a1a1a')
    gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1.5, 3, 2, 4, 0.5])

    ax0 = plt.subplot(gs[0, :])
    ax0.axis('off')
    ax0.text(0.5, 0.8, 'AgriVision-AI — Final ML Evaluation', ha='center', va='center', fontsize=36, fontweight='bold', color='white')

    model_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())
    
    model_text = f"""
Model: MobileNetV3-Small (nova_mobilenet_v3_34_classes.pth)
Classes: 34 | Input: 224 x 224
Inference: 5-view TTA (Resize 256 -> CenterCrop/4 Corners 224)
Model Size: {model_size_mb:.2f} MB | Parameters: {param_count:,}
"""
    dataset_text = f"""
Dataset Splits Evaluated
Lab Test Size = {len(lab_test)}
Field Test Size = {len(field_test)}
"""
    ax0.text(0.1, 0.3, "MODEL\n" + model_text, ha='left', va='top', fontsize=18, color='lightblue', bbox=dict(facecolor='#2a2a2a', alpha=0.5, boxstyle='round,pad=1'))
    ax0.text(0.9, 0.3, "DATASET\n" + dataset_text, ha='right', va='top', fontsize=18, color='lightgreen', bbox=dict(facecolor='#2a2a2a', alpha=0.5, boxstyle='round,pad=1'))

    ax1 = plt.subplot(gs[1, 0])
    ax1.axis('off')
    lab_info = f"""LAB METRICS
Accuracy: {lab_metrics['accuracy']:.2%}
Macro F1: {lab_metrics['macro_f1']:.2%}
Weighted F1: {lab_metrics['weighted_f1']:.2%}
"""
    ax1.text(0.5, 0.5, lab_info, ha='center', va='center', fontsize=24, color='white', bbox=dict(facecolor='#2c3e50', alpha=0.8, boxstyle='round,pad=1'))

    ax2 = plt.subplot(gs[1, 1])
    ax2.axis('off')
    field_info = f"""FIELD METRICS
Accuracy: {field_metrics['accuracy']:.2%}
Macro F1: {field_metrics['macro_f1']:.2%}
Weighted F1: {field_metrics['weighted_f1']:.2%}
Top-3 Accuracy: {field_metrics['top3_acc']:.2%}
"""
    ax2.text(0.5, 0.5, field_info, ha='center', va='center', fontsize=24, color='white', bbox=dict(facecolor='#8e44ad', alpha=0.8, boxstyle='round,pad=1'))

    ax3 = plt.subplot(gs[2, :])
    ax3.axis('off')
    ax3.text(0.5, 1.0, 'Per-Class Field Performance (Sorted by F1)', ha='center', va='top', fontsize=24, fontweight='bold', color='white')

    cell_text = []
    for row in per_class_list:
        cell_text.append([row[0], str(int(row[1])), f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"])
    table = ax3.table(cellText=cell_text, colLabels=['Class', 'Support', 'Precision', 'Recall', 'F1 Score'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('lightgray')
        else:
            cell.set_facecolor('#333333')
            cell.set_text_props(color='white')

    ax4 = plt.subplot(gs[3, 0])
    ax4.axis('off')
    ax4.text(0.5, 0.9, 'Top 10 Field Confusion Pairs', ha='center', va='top', fontsize=22, fontweight='bold', color='white')
    conf_text = "Actual -> Predicted : Count\n\n"
    for c in confusions:
        conf_text += f"{c[0]}  ->  {c[1]} : {c[2]}\n"
    if not confusions:
        conf_text += "No confusions found."
    ax4.text(0.5, 0.4, conf_text, ha='center', va='center', fontsize=14, color='white', family='monospace')

    ax5 = plt.subplot(gs[3, 1])
    ax5.axis('off')
    perf_text = f"""INFERENCE PERFORMANCE
Average Latency (TTA): {field_metrics['latency_ms']:.2f} ms / image
"""
    ax5.text(0.5, 0.5, perf_text, ha='center', va='center', fontsize=22, color='white', bbox=dict(facecolor='#d35400', alpha=0.8, boxstyle='round,pad=1'))

    ax6 = plt.subplot(gs[4, :])
    sns.heatmap(field_cm, annot=False, cmap='magma', cbar=True, ax=ax6, xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='#1a1a1a')
    ax6.set_title('34-Class Field Confusion Matrix', fontsize=24, color='white', pad=20)
    ax6.set_xlabel('Predicted Label', fontsize=16, color='white')
    ax6.set_ylabel('True Label', fontsize=16, color='white')
    ax6.tick_params(axis='x', colors='white', rotation=90, labelsize=8)
    ax6.tick_params(axis='y', colors='white', rotation=0, labelsize=8)

    ax7 = plt.subplot(gs[5, :])
    ax7.axis('off')
    ax7.text(0.5, 0.5, 'All metrics generated from the current production checkpoint and authoritative evaluation artifacts.', ha='center', va='center', fontsize=14, color='gray', style='italic')

    plt.tight_layout()
    os.makedirs('evaluation', exist_ok=True)
    plt.savefig('evaluation/final_ml_evaluation.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    print("Image saved successfully.")

if __name__ == "__main__":
    main()
