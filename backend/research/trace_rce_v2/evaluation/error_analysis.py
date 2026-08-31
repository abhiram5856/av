import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Tuple

# Add root directory to sys path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.dataset.novarcd_dataset import create_novarcd_dataloaders, deserialize_context
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.explainability.feature_importance import compute_integrated_gradients, FLATTENED_FEATURE_NAMES
from backend.research.root_cause_engine.dataset import EvalScenario

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints\best_novarcd_model.pt"

def main():
    print("========================================")
    print("NOVA-RCD: Failure Analysis & Integrated Gradients Audit")
    print("========================================")
    
    # 1. Load splits
    _, _, test_loader = create_novarcd_dataloaders(batch_size=1, augmentation_factor=0)
    
    # 2. Load model
    model = build_model()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    failures = []
    confusion_matrix = np.zeros((len(CAUSE_IDS), len(CAUSE_IDS)), dtype=int)
    
    print("Evaluating test set to identify failure cases...")
    for idx, batch in enumerate(test_loader):
        inputs = {
            k: batch[k].to(DEVICE)
            for k in ["visual", "env", "severity", "knowledge", "historical"]
        }
        
        # Ground truths
        ranks = batch["cause_ranking"][0]
        gt_indices = (ranks <= 2).nonzero(as_tuple=True)[0].tolist()
        gt_cause_ids = [CAUSE_IDS[i] for i in gt_indices]
        
        with torch.no_grad():
            outputs = model(inputs, return_attention=True)
            scores = outputs["scores"][0].cpu().numpy()
            confidences = outputs["confidences"][0].cpu().numpy()
            
        # Get sorted predictions
        pred_indices = np.argsort(scores)[::-1]
        top_pred_idx = pred_indices[0]
        top_pred_cause = CAUSE_IDS[top_pred_idx]
        top_pred_conf = confidences[top_pred_idx]
        
        if gt_cause_ids:
            primary_gt_idx = gt_indices[0]
            confusion_matrix[primary_gt_idx, top_pred_idx] += 1
            is_correct = top_pred_cause in gt_cause_ids
        else:
            primary_gt_idx = -1
            is_correct = False # Healthy plant: predicting any cause is technically a false positive
            
        if not is_correct:
            failures.append({
                "sample_idx": idx,
                "case_id": batch["case_id"][0],
                "disease_family": batch["disease_family"][0],
                "ground_truth_causes": gt_cause_ids,
                "predicted_cause": top_pred_cause,
                "predicted_confidence": float(top_pred_conf),
                "true_primary_idx": int(primary_gt_idx),
                "pred_primary_idx": int(top_pred_idx),
                "raw_batch": batch
            })
            
    print(f"Found {len(failures)} failure cases (including {sum(1 for f in failures if not f['ground_truth_causes'])} healthy/empty-GT cases) out of {len(test_loader.dataset)} test samples.")
    
    # Sort failures by confidence descending to get "confident mistakes"
    failures.sort(key=lambda x: x["predicted_confidence"], reverse=True)
    top_failures = failures[:20]
    
    print("\nComputing Integrated Gradients on top 20 confident failure cases...")
    failure_reports = []
    
    for rank, f in enumerate(top_failures):
        sample_idx = 0
        
        # Prepare inputs on DEVICE
        inputs_device = {
            k: f["raw_batch"][k].to(DEVICE)
            for k in ["visual", "env", "severity", "knowledge", "historical"]
        }
        
        # Run IG for the predicted (incorrect) cause
        ig_pred = compute_integrated_gradients(
            model=model,
            batch=inputs_device,
            sample_idx=sample_idx,
            target_cause_idx=f["pred_primary_idx"],
            steps=50
        )
        
        ig_pred_dict = {name: float(val) for name, val in zip(FLATTENED_FEATURE_NAMES, ig_pred)}
        top_contrib_pred = sorted(ig_pred_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Run IG for actual if it exists
        if f["true_primary_idx"] != -1:
            ig_true = compute_integrated_gradients(
                model=model,
                batch=inputs_device,
                sample_idx=sample_idx,
                target_cause_idx=f["true_primary_idx"],
                steps=50
            )
            ig_true_dict = {name: float(val) for name, val in zip(FLATTENED_FEATURE_NAMES, ig_true)}
        else:
            ig_true_dict = {}
            
        failure_reports.append({
            "rank": rank + 1,
            "case_id": f["case_id"],
            "disease_family": f["disease_family"],
            "predicted_cause": f["predicted_cause"],
            "predicted_confidence": f["predicted_confidence"],
            "ground_truth_causes": f["ground_truth_causes"],
            "top_contributing_features_incorrect_pred": top_contrib_pred,
            "ig_attributions_incorrect_pred": ig_pred_dict,
            "ig_attributions_correct_pred": ig_true_dict
        })
        
    # Save failure analysis
    save_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\evaluation\novarcd_failure_analysis.json"
    with open(save_path, "w") as f:
        json.dump({
            "confusion_matrix": confusion_matrix.tolist(),
            "cause_ids": CAUSE_IDS,
            "top_20_failures": failure_reports
        }, f, indent=2)
        
    print(f"Failure analysis saved to {save_path}")

if __name__ == "__main__":
    main()
