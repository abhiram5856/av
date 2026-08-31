import os
import sys
import json
import time
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v3.dataset.novarcd_v3_dataset import create_novarcd_v3_dataloaders
from backend.research.trace_rce_v3.models.trace_rce_v3 import TRACERCEv3
from backend.research.root_cause_engine.metrics import compute_metrics_package, compute_calibration_errors, compute_brier_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v3\checkpoints\best_novarcd_model.pt"
V2_METRICS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\evaluation\novarcd_scientific_eval.json"

def main():
    print("====================================================")
    print("TRACE-RCE v3: Scientific Evaluation & Comparison")
    print("====================================================")
    
    # Load dataloader
    _, _, test_loader = create_novarcd_v3_dataloaders(batch_size=32, augmentation_factor=0)
    
    # Initialize and load model
    model = TRACERCEv3()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_gts = []
    all_confs = []
    all_outcomes = []
    latencies = []
    
    # ECE Evaluation parameters
    correct_states = 0
    total_states = 0
    eas_sum = 0.0
    
    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "visual": batch["visual"].to(DEVICE),
                "env": batch["env"].to(DEVICE),
                "historical": batch["historical"].to(DEVICE),
                "knowledge": batch["knowledge"].to(DEVICE)
            }
            batch_gts = batch["binary_labels"]
            batch_ranks = batch["cause_ranking"]
            target_states = batch["reasoning_state"]
            
            # Predict sample-by-sample for latency measurement
            for b in range(batch_ranks.size(0)):
                single_inputs = {k: inputs[k][b:b+1] for k in inputs}
                
                t_start = time.perf_counter()
                outputs = model(single_inputs)
                latency = (time.perf_counter() - t_start) * 1000.0
                latencies.append(latency)
                
                scores_b = outputs["scores"][0].cpu()
                confs_b = outputs["confidences"][0].cpu()
                
                # Ranking targets
                sorted_indices = torch.argsort(scores_b, descending=True)
                pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                all_preds.append(pred_cause_ids)
                
                gt_indices = (batch_ranks[b] <= 2).nonzero(as_tuple=True)[0]
                gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                all_gts.append(gt_cause_ids)
                
                confs_list = []
                outcomes_list = []
                for idx in sorted_indices:
                    confs_list.append(confs_b[idx.item()].item())
                    outcomes_list.append(1 if CAUSE_IDS[idx.item()] in gt_cause_ids else 0)
                all_confs.append(confs_list)
                all_outcomes.append(outcomes_list)
                
                # ECE specific checks
                pred_state = torch.argmax(outputs["reasoning_state"][0]).item()
                target_state = target_states[b].item()
                if pred_state == target_state:
                    correct_states += 1
                total_states += 1
                
                # EAS (Evidence Agreement Score) is the ECE agreement score for matching inputs
                eas_sum += outputs["agreement_score"][0].item()

    # Metrics computation
    flat_confs = [c for sub in all_confs for c in sub]
    flat_outcomes = [o for sub in all_outcomes for o in sub]
    
    metrics = compute_metrics_package(all_preds, all_gts, all_confs, all_outcomes, latencies)
    ece, _ = compute_calibration_errors(flat_confs, flat_outcomes, n_bins=10)
    
    cra = correct_states / total_states if total_states > 0 else 0.0
    eas = eas_sum / total_states if total_states > 0 else 0.0
    
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        
    print("\n[TRACE-RCE v3 Performance Metrics]")
    print(f"  Precision@1 : {metrics['precision_at_1']:.4f}")
    print(f"  Precision@3 : {metrics['precision_at_3']:.4f}")
    print(f"  Recall@3    : {metrics['recall_at_3']:.4f}")
    print(f"  MRR         : {metrics['mrr']:.4f}")
    print(f"  NDCG@3      : {metrics['ndcg_at_3']:.4f}")
    print(f"  Brier Score : {metrics['brier_score']:.4f}")
    print(f"  ECE         : {ece:.4f}")
    print(f"  Conflict Resolution Accuracy (CRA): {cra:.4f}")
    print(f"  Evidence Agreement Score (EAS)   : {eas:.4f}")
    print(f"  Mean Latency: {metrics['mean']:.4f} ms")
    print(f"  Peak GPU Mem: {peak_mem_mb:.2f} MB")
    
    # 3. Compare with v2 baseline
    v2_metrics = {}
    if os.path.exists(V2_METRICS_PATH):
        with open(V2_METRICS_PATH, "r") as f:
            v2_data = json.load(f)
            v2_metrics = v2_data.get("metrics", {})
            v2_latency = v2_data.get("latency", {}).get("mean", 0.0)
            v2_mem = v2_data.get("peak_memory_mb", 0.0)
            
    print("\n[COMPARISON WITH TRACE-RCE v2 BASELINE]")
    if v2_metrics:
        print(f"  NDCG@3      : v3 = {metrics['ndcg_at_3']:.4f} vs v2 = {v2_metrics.get('ndcg_at_3', 0.0):.4f}")
        print(f"  Precision@1 : v3 = {metrics['precision_at_1']:.4f} vs v2 = {v2_metrics.get('precision_at_1', 0.0):.4f}")
        print(f"  Recall@3    : v3 = {metrics['recall_at_3']:.4f} vs v2 = {v2_metrics.get('recall_at_3', 0.0):.4f}")
        print(f"  Brier Score : v3 = {metrics['brier_score']:.4f} vs v2 = {v2_metrics.get('brier_score', 0.0):.4f}")
        print(f"  ECE         : v3 = {ece:.4f} vs v2 = {v2_metrics.get('ece', 0.0):.4f}")
        print(f"  Mean Latency: v3 = {metrics['mean']:.4f} ms vs v2 = {v2_latency:.4f} ms")
        print(f"  Peak GPU Mem: v3 = {peak_mem_mb:.2f} MB vs v2 = {v2_mem:.2f} MB")
    else:
        print("  v2 scientific baseline metrics not found.")
        
    # Save comparison data to JSON
    out_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v3\evaluation\trace_rce_v3_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "v3_metrics": {
                "precision_at_1": metrics["precision_at_1"],
                "precision_at_3": metrics["precision_at_3"],
                "recall_at_3": metrics["recall_at_3"],
                "mrr": metrics["mrr"],
                "ndcg_at_3": metrics["ndcg_at_3"],
                "brier_score": metrics["brier_score"],
                "ece": ece,
                "cra": cra,
                "eas": eas
            },
            "v3_latency": metrics["mean"],
            "v3_peak_memory_mb": peak_mem_mb,
            "v2_comparison": v2_metrics
        }, f, indent=2)
    print(f"\nEvaluation results saved to {out_path}")

if __name__ == "__main__":
    main()
