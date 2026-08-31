import os
import sys
import json
import time
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.dataset.novarcd_dataset import create_novarcd_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.root_cause_engine.metrics import compute_metrics_package, compute_calibration_errors, compute_ndcg_at_k, compute_brier_score, compute_precision_at_k, compute_recall_at_k, compute_mrr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints\best_novarcd_model.pt"

def bootstrap_confidence_intervals(preds, gts, confs, outcomes, n_resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = len(preds)
    
    metrics_boot = {
        "precision_at_1": [],
        "precision_at_3": [],
        "recall_at_3": [],
        "mrr": [],
        "ndcg_at_3": [],
        "brier_score": [],
        "ece": []
    }
    
    # Precompute per-sample metrics for speed
    p1_list = [compute_precision_at_k(p, gt, 1) for p, gt in zip(preds, gts)]
    p3_list = [compute_precision_at_k(p, gt, 3) for p, gt in zip(preds, gts)]
    r3_list = [compute_recall_at_k(p, gt, 3) for p, gt in zip(preds, gts)]
    mrr_list = [compute_mrr(p, gt) for p, gt in zip(preds, gts)]
    ndcg_list = [compute_ndcg_at_k(p, gt, 3) for p, gt in zip(preds, gts)]
    
    brier_list = [compute_brier_score(c, o) for c, o in zip(confs, outcomes)]
    
    # ECE requires all predictions flat, so we compute ECE inside the loop over the sampled subset
    for _ in range(n_resamples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        metrics_boot["precision_at_1"].append(np.mean([p1_list[idx] for idx in indices]))
        metrics_boot["precision_at_3"].append(np.mean([p3_list[idx] for idx in indices]))
        metrics_boot["recall_at_3"].append(np.mean([r3_list[idx] for idx in indices]))
        metrics_boot["mrr"].append(np.mean([mrr_list[idx] for idx in indices]))
        metrics_boot["ndcg_at_3"].append(np.mean([ndcg_list[idx] for idx in indices]))
        metrics_boot["brier_score"].append(np.mean([brier_list[idx] for idx in indices]))
        
        # Compute ECE on this resampled set
        flat_confs_b = []
        flat_outcomes_b = []
        for idx in indices:
            flat_confs_b.extend(confs[idx])
            flat_outcomes_b.extend(outcomes[idx])
        
        ece_b, _ = compute_calibration_errors(flat_confs_b, flat_outcomes_b, n_bins=10)
        metrics_boot["ece"].append(ece_b)
        
    ci = {}
    for k, values in metrics_boot.items():
        sorted_val = sorted(values)
        lower = sorted_val[int(0.025 * n_resamples)]
        upper = sorted_val[int(0.975 * n_resamples)]
        ci[k] = (float(lower), float(upper))
        
    return ci

def main():
    print("========================================")
    print("NOVA-RCD: Scientific Evaluation of TRACE-RCE v2")
    print("========================================")
    
    _, _, test_loader = create_novarcd_dataloaders(batch_size=32, augmentation_factor=0)
    
    model = build_model()
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
    
    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        
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
                
                t_start = time.perf_counter()
                outputs = model(single_inputs, return_attention=False)
                latency = (time.perf_counter() - t_start) * 1000.0 # ms
                latencies.append(latency)
                
                scores_b = outputs["scores"][0].cpu()
                confs_b = outputs["confidences"][0].cpu()
                
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
                
    # Flat arrays for calibration error
    flat_confs = [c for sub in all_confs for c in sub]
    flat_outcomes = [o for sub in all_outcomes for o in sub]
    
    metrics = compute_metrics_package(all_preds, all_gts, all_confs, all_outcomes, latencies)
    ece, _ = compute_calibration_errors(flat_confs, flat_outcomes, n_bins=10)
    
    # Peak memory allocated
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)
        
    print("\n[Point Estimates]")
    print(f"Precision@1 : {metrics['precision_at_1']:.4f}")
    print(f"Precision@3 : {metrics['precision_at_3']:.4f}")
    print(f"Recall@3    : {metrics['recall_at_3']:.4f}")
    print(f"MRR         : {metrics['mrr']:.4f}")
    print(f"NDCG@3      : {metrics['ndcg_at_3']:.4f}")
    print(f"Brier Score : {metrics['brier_score']:.4f}")
    print(f"ECE         : {ece:.4f}")
    
    print("\n[Latency (ms)]")
    print(f"Mean        : {metrics['mean']:.4f}")
    print(f"p50         : {metrics['p50']:.4f}")
    print(f"p95         : {metrics['p95']:.4f}")
    print(f"p99         : {metrics['p99']:.4f}")
    
    print(f"\nPeak CUDA Memory: {peak_mem_mb:.2f} MB")
    
    # Compute Bootstrap Confidence Intervals (95% CI)
    print("\nRunning Bootstrap Significance testing (B=1000 resamples)...")
    ci = bootstrap_confidence_intervals(all_preds, all_gts, all_confs, all_outcomes, n_resamples=1000)
    
    print("\n[95% Bootstrap Confidence Intervals]")
    for k, (lower, upper) in ci.items():
        print(f"{k:<15} : [{lower:.4f}, {upper:.4f}]")
        
    # Save results to json
    out_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\evaluation\novarcd_scientific_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "metrics": {
                "precision_at_1": metrics["precision_at_1"],
                "precision_at_3": metrics["precision_at_3"],
                "recall_at_3": metrics["recall_at_3"],
                "mrr": metrics["mrr"],
                "ndcg_at_3": metrics["ndcg_at_3"],
                "brier_score": metrics["brier_score"],
                "ece": ece
            },
            "latency": {
                "mean": metrics["mean"],
                "p50": metrics["p50"],
                "p95": metrics["p95"],
                "p99": metrics["p99"]
            },
            "peak_memory_mb": peak_mem_mb,
            "ci": ci
        }, f, indent=2)
    print(f"\nScientific evaluation results saved to {out_path}")

if __name__ == "__main__":
    main()
