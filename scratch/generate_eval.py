import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec

# Load metrics
with open('evaluation/final_field_metrics.json', 'r') as f:
    field_metrics = json.load(f)

with open('evaluation/final_lab_metrics.json', 'r') as f:
    lab_metrics = json.load(f)

with open('evaluation/dataset_audit_report.json', 'r') as f:
    dataset = json.load(f)

classes = dataset['classes']
field_cm = np.array(field_metrics['confusion_matrix'])
field_per_class = field_metrics['per_class']

# Calculate top 10 confusion pairs
confusions = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j:
            count = field_cm[i][j]
            if count > 0:
                confusions.append((classes[i], classes[j], count))
confusions = sorted(confusions, key=lambda x: x[2], reverse=True)[:10]

# Prepare per-class F1 for table
per_class_list = []
for c in classes:
    m = field_per_class.get(c, {'precision': 0, 'recall': 0, 'f1_score': 0, 'support': 0})
    per_class_list.append((c, m['support'], m['precision'], m['recall'], m['f1_score']))

per_class_list = sorted(per_class_list, key=lambda x: x[4])

# Start plotting
plt.style.use('dark_background')
fig = plt.figure(figsize=(24, 30))
fig.patch.set_facecolor('#1a1a1a')
gs = gridspec.GridSpec(6, 2, height_ratios=[1, 1.5, 3, 2, 4, 0.5])

# 1. Title and Model & Dataset
ax0 = plt.subplot(gs[0, :])
ax0.axis('off')
ax0.text(0.5, 0.8, 'AgriVision-AI — Final ML Evaluation', ha='center', va='center', fontsize=36, fontweight='bold', color='white')

model_text = """
Model: MobileNetV3-Small (nova_mobilenet_v3_34_classes.pth)
Classes: 34 | Input: 224 x 224
Inference: 5-view TTA
"""
dataset_text = f"""
Dataset: Lab Total = {dataset['total_images']} | Lab Train = {dataset['splits']['train']} | Lab Val = {dataset['splits']['val']} | Lab Test = {dataset['splits']['test']}
Field Test = {field_metrics['total_samples']} | Number of classes = {dataset['total_classes']}
"""
ax0.text(0.1, 0.3, "MODEL\n" + model_text, ha='left', va='top', fontsize=18, color='lightblue', bbox=dict(facecolor='#2a2a2a', alpha=0.5, boxstyle='round,pad=1'))
ax0.text(0.9, 0.3, "DATASET\n" + dataset_text, ha='right', va='top', fontsize=18, color='lightgreen', bbox=dict(facecolor='#2a2a2a', alpha=0.5, boxstyle='round,pad=1'))

# 2. Lab & Field Performance Summary Cards
ax1 = plt.subplot(gs[1, 0])
ax1.axis('off')
lab_info = f"""LAB METRICS
Accuracy: {lab_metrics['accuracy']:.2%}
Macro Precision: {lab_metrics['macro_precision']:.2%}
Macro Recall: {lab_metrics['macro_recall']:.2%}
Macro F1: {lab_metrics['macro_f1']:.2%}
Weighted F1: {lab_metrics['weighted_f1']:.2%}
"""
ax1.text(0.5, 0.5, lab_info, ha='center', va='center', fontsize=20, color='white', bbox=dict(facecolor='#2c3e50', alpha=0.8, boxstyle='round,pad=1'))

ax2 = plt.subplot(gs[1, 1])
ax2.axis('off')
field_info = f"""FIELD METRICS
Accuracy: {field_metrics['accuracy']:.2%}
Top-3 Accuracy: {field_metrics['top3_accuracy']:.2%}
Macro Precision: {field_metrics['macro_precision']:.2%}
Macro Recall: {field_metrics['macro_recall']:.2%}
Macro F1: {field_metrics['macro_f1']:.2%}
Weighted F1: {field_metrics['weighted_f1']:.2%}
"""
ax2.text(0.5, 0.5, field_info, ha='center', va='center', fontsize=20, color='white', bbox=dict(facecolor='#8e44ad', alpha=0.8, boxstyle='round,pad=1'))

# 3. Class Level Results Table (Compact)
ax3 = plt.subplot(gs[2, :])
ax3.axis('off')
ax3.text(0.5, 1.0, 'Per-Class Field Performance (Sorted by F1)', ha='center', va='top', fontsize=24, fontweight='bold', color='white')

# create table
cell_text = []
for row in per_class_list:
    cell_text.append([row[0], str(row[1]), f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"])
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

# 4. Top Confusions
ax4 = plt.subplot(gs[3, 0])
ax4.axis('off')
ax4.text(0.5, 0.9, 'Top 10 Field Confusion Pairs', ha='center', va='top', fontsize=22, fontweight='bold', color='white')
conf_text = "Actual -> Predicted : Count\n\n"
for c in confusions:
    conf_text += f"{c[0]}  ->  {c[1]} : {c[2]}\n"
if not confusions:
    conf_text += "No confusions found."
ax4.text(0.5, 0.4, conf_text, ha='center', va='center', fontsize=14, color='white', family='monospace')

# 5. Performance and Final Note
ax5 = plt.subplot(gs[3, 1])
ax5.axis('off')
perf_text = f"""INFERENCE PERFORMANCE
Average Latency: {field_metrics['latency_ms_per_image']:.2f} ms / image

EXPERIMENT RESULT
candidate_v3: REJECTED — preprocessing mismatch
"""
ax5.text(0.5, 0.5, perf_text, ha='center', va='center', fontsize=18, color='white', bbox=dict(facecolor='#d35400', alpha=0.8, boxstyle='round,pad=1'))

# 6. Confusion Matrix
ax6 = plt.subplot(gs[4, :])
sns.heatmap(field_cm, annot=False, cmap='magma', cbar=True, ax=ax6, xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='#1a1a1a')
ax6.set_title('34-Class Field Confusion Matrix', fontsize=24, color='white', pad=20)
ax6.set_xlabel('Predicted Label', fontsize=16, color='white')
ax6.set_ylabel('True Label', fontsize=16, color='white')
ax6.tick_params(axis='x', colors='white', rotation=90, labelsize=8)
ax6.tick_params(axis='y', colors='white', rotation=0, labelsize=8)

# 7. Footer
ax7 = plt.subplot(gs[5, :])
ax7.axis('off')
ax7.text(0.5, 0.5, 'All metrics generated from the current production checkpoint and authoritative evaluation artifacts.', ha='center', va='center', fontsize=14, color='gray', style='italic')

plt.tight_layout()
os.makedirs('evaluation', exist_ok=True)
plt.savefig('evaluation/final_ml_evaluation.png', facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
print("Image saved successfully.")
