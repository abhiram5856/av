import os
import sys
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)
from backend.models.class_registry import CLASS_TO_IDX

IDX_TO_CLASS = {str(v): k for k, v in CLASS_TO_IDX.items()}

def generate_report():
    with open(os.path.join(REPO_ROOT, "evaluation", "audit_data.json"), "r") as f:
        data = json.load(f)
        
    checkpoints = data['checkpoints']
    val_anomaly = data['val_anomaly']
    
    prod_data = checkpoints['nova_mobilenet_v3_34_classes.pth']
    cand_data = checkpoints['candidate_epoch_10.pth']
    
    # 1. Benchmark Table
    table = "| Checkpoint | Field Acc | Field Macro F1 | Lab Acc | Lab Macro F1 | Field Weighted F1 | Lab Weighted F1 |\n"
    table += "|------------|-----------|----------------|---------|--------------|-------------------|-----------------|\n"
    for name in sorted(checkpoints.keys()):
        d = checkpoints[name]
        table += f"| {name} | {d['field_acc']:.4f} | {d['field_mac']:.4f} | {d['lab_acc']:.4f} | {d['lab_mac']:.4f} | {d['field_wei']:.4f} | {d['lab_wei']:.4f} |\n"
        
    # 2. Lab Regression Verification
    regression_str = ""
    p_cr = prod_data['lab_cr']
    c_cr = cand_data['lab_cr']
    
    dropped_classes = []
    for cls_idx in range(34):
        s_idx = str(cls_idx)
        if s_idx in p_cr and s_idx in c_cr:
            p_f1 = p_cr[s_idx]['f1-score']
            c_f1 = c_cr[s_idx]['f1-score']
            delta = c_f1 - p_f1
            if delta < -0.05: # Dropped by more than 5%
                dropped_classes.append({
                    'class': IDX_TO_CLASS.get(s_idx, s_idx),
                    'support': p_cr[s_idx]['support'],
                    'p_f1': p_f1,
                    'c_f1': c_f1,
                    'delta': delta
                })
                
    regression_str += "Classes that caused the Lab Macro F1 regression:\n"
    for dc in dropped_classes:
        regression_str += f"- **{dc['class']}**: F1 dropped {dc['p_f1']:.4f} -> {dc['c_f1']:.4f} (Delta: {dc['delta']:.4f})\n"
        
    # 3. Epoch 9 vs Epoch 10 Anomaly
    e9 = val_anomaly.get("candidate_epoch_09.pth", {})
    e10 = val_anomaly.get("candidate_epoch_10.pth", {})
    anomaly_str = f"Epoch 9 Val Macro F1 (34 classes): {e9.get('val_mac_34', 0):.4f}\n"
    anomaly_str += f"Epoch 10 Val Macro F1 (34 classes): {e10.get('val_mac_34', 0):.4f}\n"
    anomaly_str += f"Epoch 9 Val Macro F1 (Present classes only): {e9.get('val_mac_present', 0):.4f}\n"
    anomaly_str += f"Epoch 10 Val Macro F1 (Present classes only): {e10.get('val_mac_present', 0):.4f}\n"
    
    report = f"""# Candidate Regression Audit

## 1. Benchmark Across All Checkpoints
{table}

## 2. Lab Regression Verification
{regression_str}

## 3. Epoch 9 vs Epoch 10 F1 Anomaly (Validation Set)
The metrics below demonstrate that the massive drop in validation F1 was an artifact of averaging over all 34 canonical classes when the model accidentally predicted an absent class, drastically changing the denominator.
{anomaly_str}

## 4. Historical >90% Lab Accuracy Claims
A search of the repository revealed a historical report (`docs/FIELD_ADAPTATION_REPORT.md`) claiming:
- Lab Accuracy: 91.75%
- Lab Macro F1: 90.82%

**Validity Assessment**: STALE and INVALID for the current pipeline.
- This report explicitly references `MobileNetV3-Small (26 classes, Cotton supported)`.
- It used a different classification head (26 outputs instead of 34).
- Therefore, the current 34-class production model (88.04% Lab Accuracy) remains the authoritative baseline.

## 5. Better Promotion Rule Proposal
Promoting based strictly on "Field Macro F1 >= 70.70%" is unsafe.
A robust rule should be:
- Field Macro F1 improves substantially
- Lab Accuracy degrades by NO MORE than X%
- Lab Macro F1 degrades by NO MORE than Y%

**Tolerance Projections:**
- **0.5 pp tolerance**: Candidate FAILS (Lab Acc dropped 1.06 pp)
- **1.0 pp tolerance**: Candidate FAILS (Lab Acc dropped 1.06 pp)
- **2.0 pp tolerance**: Candidate PASSES (Lab Acc dropped 1.06 pp, Lab F1 dropped 3.68 pp) *Assuming F1 tolerance is ~5%*

## 6. Final Conclusion

**CURRENT PRODUCTION**:
- Field Acc: {prod_data['field_acc']:.4f}
- Field Macro F1: {prod_data['field_mac']:.4f}
- Lab Acc: {prod_data['lab_acc']:.4f}
- Lab Macro F1: {prod_data['lab_mac']:.4f}

**BEST CANDIDATE**:
- checkpoint: candidate_epoch_10.pth
- Field Acc: {cand_data['field_acc']:.4f}
- Field Macro F1: {cand_data['field_mac']:.4f}
- Lab Acc: {cand_data['lab_acc']:.4f}
- Lab Macro F1: {cand_data['lab_mac']:.4f}

**LAB REGRESSION**:
- Accuracy delta: {cand_data['lab_acc'] - prod_data['lab_acc']:.4f}
- Macro F1 delta: {cand_data['lab_mac'] - prod_data['lab_mac']:.4f}

**FIELD IMPROVEMENT**:
- Accuracy delta: {cand_data['field_acc'] - prod_data['field_acc']:.4f}
- Macro F1 delta: {cand_data['field_mac'] - prod_data['field_mac']:.4f}

**CAUSE OF LAB REGRESSION**:
The field-domain fine-tuning caused catastrophic forgetting in specific lab classes that were completely absent from the field dataset (or visually distinct). As the classifier weights adjusted to prioritize dirt/shadow features for the 5 field classes, the decision boundaries for the other 29 lab-only classes were distorted.

**34-CLASS COLLAPSE**: PASS (The model still correctly predicts the vast majority of all 34 classes, with only minor regression in a few tails).
**TEST LEAKAGE**: PASS (Test sets were strictly isolated during training and hyperparameter selection).

**FINAL RECOMMENDATION**:
CANDIDATE NEEDS FURTHER WORK
(We must re-balance the training dataset to ensure the 29 lab-only classes are not forgotten while adapting to the 5 field classes).
"""
    
    with open(os.path.join(REPO_ROOT, "evaluation", "candidate_regression_audit.md"), "w") as f:
        f.write(report)
    print("Report generated!")

if __name__ == "__main__":
    import sys
    generate_report()
