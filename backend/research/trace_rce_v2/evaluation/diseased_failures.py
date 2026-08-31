import os
import sys
import json
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.dataset.novarcd_dataset import create_novarcd_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints\best_novarcd_model.pt"

def main():
    _, _, test_loader = create_novarcd_dataloaders(batch_size=32, augmentation_factor=0)
    
    model = build_model()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    failures = []
    successes = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                k: batch[k].to(DEVICE)
                for k in ["visual", "env", "severity", "knowledge", "historical"]
            }
            batch_gts = batch["binary_labels"]
            batch_ranks = batch["cause_ranking"]
            
            for b in range(batch_ranks.size(0)):
                single_inputs = {k: inputs[k][b:b+1] for k in inputs}
                outputs = model(single_inputs, return_attention=False)
                
                scores_b = outputs["scores"][0].cpu()
                confs_b = outputs["confidences"][0].cpu()
                
                sorted_indices = torch.argsort(scores_b, descending=True)
                pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                
                gt_indices = (batch_ranks[b] <= 2).nonzero(as_tuple=True)[0]
                gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                
                if len(gt_cause_ids) > 0:
                    primary_gt = gt_cause_ids[0]
                    primary_pred = pred_cause_ids[0]
                    confidence = confs_b[sorted_indices[0].item()].item()
                    
                    record_info = {
                        "case_id": batch["case_id"][b],
                        "disease_family": batch["disease_family"][b],
                        "primary_gt": primary_gt,
                        "primary_pred": primary_pred,
                        "confidence": confidence,
                        "ground_truth_causes": gt_cause_ids,
                        "predicted_causes": pred_cause_ids[:3]
                    }
                    
                    if primary_pred != primary_gt:
                        failures.append(record_info)
                    else:
                        successes.append(record_info)
                        
    print(f"Total Diseased Cases evaluated: {len(failures) + len(successes)}")
    print(f"Correct primary cause predictions: {len(successes)}")
    print(f"Incorrect primary cause predictions: {len(failures)}")
    
    # Sort failures by confidence (descending) to find top confident failures
    failures.sort(key=lambda x: x["confidence"], reverse=True)
    
    print("\n[TOP 10 CONFIDENT FAILURES IN DISEASED CASES]")
    for idx, f in enumerate(failures[:10]):
        print(f"{idx+1}. Case: {f['case_id']} | Family: {f['disease_family']}")
        print(f"   GT Primary:   {f['primary_gt']}")
        print(f"   Pred Primary: {f['primary_pred']} (Conf: {f['confidence']:.4f})")
        print(f"   GT Causes:    {f['ground_truth_causes']}")
        print(f"   Pred top-3:   {f['predicted_causes']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
