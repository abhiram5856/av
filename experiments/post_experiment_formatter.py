import os
import json
import csv
import numpy as np

BASE_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
FIELD_IMP_DIR = os.path.join(EVAL_DIR, "field_improvement")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

def format_all_artifacts():
    # Load baseline metrics
    baseline_path = os.path.join(FIELD_IMP_DIR, "baseline_metrics.json")
    if not os.path.exists(baseline_path):
        print("Baseline metrics not found!")
        return

    with open(baseline_path, 'r') as f:
        base_data = json.load(f)

    # 1. Save evaluation/baseline_lab_metrics.json
    with open(os.path.join(EVAL_DIR, "baseline_lab_metrics.json"), 'w') as f:
        json.dump(base_data["lab_test"], f, indent=2)

    # 2. Save evaluation/baseline_field_metrics.json
    with open(os.path.join(EVAL_DIR, "baseline_field_metrics.json"), 'w') as f:
        json.dump(base_data["field_test"], f, indent=2)

    # 3. Save evaluation/baseline_confusion_matrix.json
    with open(os.path.join(EVAL_DIR, "baseline_confusion_matrix.json"), 'w') as f:
        json.dump(base_data["field_test"]["confusion_matrix"], f, indent=2)

    # 4. Save evaluation/field_error_analysis.md
    error_analysis_md = f"""# Field Image Diagnosis Error Analysis

## Baseline Evaluation Summary
- **Field Single-Image Accuracy:** {base_data['field_test']['accuracy']*100:.2f}%
- **Field Top-3 Accuracy:** {base_data['field_test']['top3_accuracy']*100:.2f}%
- **Field Macro F1:** {base_data['field_test']['macro_f1']*100:.2f}%

## Primary Root Causes of Field Failure
1. **Background Complexity:** Laboratory images feature synthetic white/black backgrounds, whereas field images contain complex soil, weed, and ambient foliage backgrounds.
2. **Class Imbalance:** Field dataset split is heavily concentrated on Rice and Cotton diseases. Unrepresented classes lower unweighted Macro F1.
3. **Lighting & Shadows:** Direct sunlight glare and shadows produce high-frequency visual noise that shifts feature representation.
4. **Visually Similar Symptoms:** Bacterial blight vs brown spot on rice leaves present similar visual necrotic spots under uncalibrated mobile camera lenses.

## Recommendations
- Retain single-source-of-truth 34-class registry (`class_registry.py`).
- Use class-weighted loss and domain-aware augmentation (crop, color jitter, blur) to generalize feature extraction across background variations.
"""
    with open(os.path.join(EVAL_DIR, "field_error_analysis.md"), 'w') as f:
        f.write(error_analysis_md)

    # Load experiment comparison results if available
    comp_path = os.path.join(FIELD_IMP_DIR, "model_comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, 'r') as f:
            comp_data = json.load(f)

        # 5. Save evaluation/domain_adaptation_experiments.csv
        csv_path = os.path.join(EVAL_DIR, "domain_adaptation_experiments.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model/Experiment", "Field Val Acc", "Field Val Macro F1", 
                "Lab Test Acc", "Lab Macro F1", "Field Test Acc", "Field Test Macro F1"
            ])
            for k, v in comp_data.items():
                if isinstance(v, dict):
                    writer.writerow([
                        k, 
                        f"{v.get('field_val_acc', 0.0)*100:.2f}%", 
                        f"{v.get('field_val_macro_f1', 0.0)*100:.2f}%",
                        f"{v.get('lab_acc', 0.0)*100:.2f}%", 
                        f"{v.get('lab_macro_f1', 0.0)*100:.2f}%",
                        f"{v.get('field_test_acc', 0.0)*100:.2f}%" if 'field_test_acc' in v else "N/A",
                        f"{v.get('field_test_macro_f1', 0.0)*100:.2f}%" if 'field_test_macro_f1' in v else "N/A"
                    ])

        # 6. Save evaluation/model_comparison.md
        md_comp_path = os.path.join(EVAL_DIR, "model_comparison.md")
        with open(md_comp_path, 'w') as f:
            f.write("# Model Comparison Table\n\n")
            f.write("| Model | Field Val Acc | Field Val Macro F1 | Lab Test Acc | Lab Macro F1 |\n")
            f.write("|---|---|---|---|---|\n")
            for k, v in comp_data.items():
                if isinstance(v, dict):
                    f.write(f"| **{k}** | {v.get('field_val_acc', 0.0)*100:.2f}% | {v.get('field_val_macro_f1', 0.0)*100:.2f}% | {v.get('lab_acc', 0.0)*100:.2f}% | {v.get('lab_macro_f1', 0.0)*100:.2f}% |\n")

    # Load final field evaluation if available
    final_eval_path = os.path.join(FIELD_IMP_DIR, "final_field_evaluation.json")
    if os.path.exists(final_eval_path):
        with open(final_eval_path, 'r') as f:
            final_data = json.load(f)

        # 7. Save evaluation/final_field_metrics.json
        with open(os.path.join(EVAL_DIR, "final_field_metrics.json"), 'w') as f:
            json.dump(final_data["field_test_metrics"], f, indent=2)

        # 8. Save evaluation/final_lab_metrics.json
        with open(os.path.join(EVAL_DIR, "final_lab_metrics.json"), 'w') as f:
            json.dump(final_data["lab_test_metrics"], f, indent=2)

        # 9. Save evaluation/final_confusion_matrix.json
        with open(os.path.join(EVAL_DIR, "final_confusion_matrix.json"), 'w') as f:
            json.dump(final_data["field_test_metrics"]["confusion_matrix"], f, indent=2)

    print("All required evaluation artifacts formatted and saved successfully.")

if __name__ == "__main__":
    format_all_artifacts()
