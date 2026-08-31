"""
TRACE-RCE v2 — Evaluation & Comparison
=========================================
Runs head-to-head evaluation of TRACE-RCE v1 (rule-based) vs v2 (trainable).
Uses the exact same test split, same metrics, and computes statistical
significance via paired bootstrap testing.

Metrics Evaluated
-----------------
1. Precision@1, Precision@3, Recall@3
2. Mean Reciprocal Rank (MRR)
3. Normalized Discounted Cumulative Gain at 3 (NDCG@3)
4. Expected Calibration Error (ECE)
5. Brier Score (Calibration quality)
6. Inference Latency (Mean, p50, p95, p99)
7. Model Size and Parameter Count

Statistical Significance
------------------------
Computes p-values for NDCG@3 and Brier Score differences using a paired
bootstrap test with B=10,000 resamples to evaluate if v2 improvements
are statistically significant (p < 0.05).
"""

import os
import time
import json
import asyncio
import numpy as np
import torch
import logging
from typing import Dict, Any, List, Tuple

from torch.utils.data import DataLoader
from backend.research.root_cause_engine.dataset import generate_evaluation_dataset, EvalScenario
from backend.research.trace_rce_v2.dataset.dataloader import stratified_split, create_dataloaders
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.root_cause_engine.engine import TRACERootCauseEngine
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.root_cause_engine.metrics import compute_metrics_package, compute_ndcg_at_k, compute_brier_score

logger = logging.getLogger("nova.research.trace_rce_v2.evaluator")

# =============================================================================
# Baseline V1 Evaluation (Async Wrapper)
# =============================================================================

async def run_v1_evaluation(test_scenarios: List[EvalScenario]) -> Tuple[Dict[str, Any], Dict[str, List]]:
    """Evaluate TRACE-RCE v1 on test split."""
    engine = TRACERootCauseEngine()
    
    all_preds: List[List[str]] = []
    all_gts: List[List[str]] = []
    all_confs: List[List[float]] = []
    all_outcomes: List[List[int]] = []
    latencies: List[float] = []

    logger.info("Evaluating TRACE-RCE v1 on %d test cases...", len(test_scenarios))

    for scenario in test_scenarios:
        t_start = time.perf_counter()
        # v1 analyze is async
        result = await engine.analyze(scenario.context)
        latency = (time.perf_counter() - t_start) * 1000.0
        latencies.append(latency)

        # Ranked list of cause IDs from result
        pred_cause_ids = [rc.cause_id for rc in result.ranked_causes]
        # Ensure we pad/handle cases where not all 13 are present
        all_preds.append(pred_cause_ids)

        gt_cause_ids = scenario.ground_truth_causes
        all_gts.append(gt_cause_ids)

        # Map confidences
        # If cause was evaluated, get its calibrated upper bound
        confs_dict = {}
        for rc in result.ranked_causes:
            confs_dict[rc.cause_id] = rc.confidence.upper
        
        # Sort confidences and outcomes in prediction order
        confs_b = []
        outcomes_b = []
        for cid in pred_cause_ids:
            confs_b.append(confs_dict.get(cid, 0.0))
            outcomes_b.append(1 if cid in gt_cause_ids else 0)
        
        # Pad up to 13 causes if some were filtered out by threshold
        remaining_causes = [cid for cid in CAUSE_IDS if cid not in pred_cause_ids]
        for cid in remaining_causes:
            confs_b.append(0.0)
            outcomes_b.append(0)
            pred_cause_ids.append(cid)

        all_confs.append(confs_b)
        all_outcomes.append(outcomes_b)

    metrics = compute_metrics_package(
        all_predictions=all_preds,
        all_ground_truths=all_gts,
        all_confidences=all_confs,
        all_outcomes=all_outcomes,
        latencies=latencies
    )

    raw_results = {
        "predictions": all_preds,
        "ground_truths": all_gts,
        "confidences": all_confs,
        "outcomes": all_outcomes,
        "latencies": latencies
    }

    return metrics, raw_results


# =============================================================================
# Trainable V2 Evaluation
# =============================================================================

def run_v2_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, Any], Dict[str, List]]:
    """Evaluate TRACE-RCE v2 model on test split."""
    model.eval()
    
    all_preds: List[List[str]] = []
    all_gts: List[List[str]] = []
    all_confs: List[List[float]] = []
    all_outcomes: List[List[int]] = []
    latencies: List[float] = []

    logger.info("Evaluating TRACE-RCE v2 on %d test cases...", len(test_loader.dataset))

    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                k: batch[k].to(device)
                for k in ["visual", "env", "severity", "knowledge", "historical"]
            }
            batch_gts = batch["binary_labels"]
            batch_ranks = batch["cause_ranking"]

            for b in range(batch_ranks.size(0)):
                # Package a single item batch for latency measurement
                single_inputs = {k: inputs[k][b:b+1] for k in inputs}
                
                t_start = time.perf_counter()
                outputs = model(single_inputs, return_attention=False)
                latency = (time.perf_counter() - t_start) * 1000.0
                latencies.append(latency)

                scores_b = outputs["scores"][0].cpu()
                confs_b = outputs["confidences"][0].cpu()

                # Get sorted indices descending
                sorted_indices = torch.argsort(scores_b, descending=True)
                pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                all_preds.append(pred_cause_ids)

                # Ground truths
                gt_indices = (batch_ranks[b] <= 2).nonzero(as_tuple=True)[0]
                gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                all_gts.append(gt_cause_ids)

                # Confidences and binary outcomes
                confs_list = []
                outcomes_list = []
                for idx in sorted_indices:
                    confs_list.append(confs_b[idx.item()].item())
                    outcomes_list.append(1 if CAUSE_IDS[idx.item()] in gt_cause_ids else 0)

                all_confs.append(confs_list)
                all_outcomes.append(outcomes_list)

    metrics = compute_metrics_package(
        all_predictions=all_preds,
        all_ground_truths=all_gts,
        all_confidences=all_confs,
        all_outcomes=all_outcomes,
        latencies=latencies
    )

    raw_results = {
        "predictions": all_preds,
        "ground_truths": all_gts,
        "confidences": all_confs,
        "outcomes": all_outcomes,
        "latencies": latencies
    }

    return metrics, raw_results


# =============================================================================
# Paired Bootstrap Significance Testing
# =============================================================================

def paired_bootstrap_test(
    v1_raw: Dict[str, List],
    v2_raw: Dict[str, List],
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Perform a non-parametric paired bootstrap significance test.
    We resample the test set index pairs with replacement, compute the difference
    in NDCG@3 and Brier score, and calculate p-values.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(v1_raw["predictions"])
    
    ndcg_diffs = []
    brier_diffs = []

    # Precompute per-sample metrics
    v1_ndcg = [compute_ndcg_at_k(p, gt, 3) for p, gt in zip(v1_raw["predictions"], v1_raw["ground_truths"])]
    v2_ndcg = [compute_ndcg_at_k(p, gt, 3) for p, gt in zip(v2_raw["predictions"], v2_raw["ground_truths"])]

    v1_brier = [compute_brier_score(c, o) for c, o in zip(v1_raw["confidences"], v1_raw["outcomes"])]
    v2_brier = [compute_brier_score(c, o) for c, o in zip(v2_raw["confidences"], v2_raw["outcomes"])]

    # Actual observed mean differences
    observed_ndcg_diff = np.mean(v2_ndcg) - np.mean(v1_ndcg)
    observed_brier_diff = np.mean(v2_brier) - np.mean(v1_brier)

    for _ in range(n_resamples):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        
        ndcg_v1_b = np.mean([v1_ndcg[idx] for idx in indices])
        ndcg_v2_b = np.mean([v2_ndcg[idx] for idx in indices])
        ndcg_diffs.append(ndcg_v2_b - ndcg_v1_b)

        brier_v1_b = np.mean([v1_brier[idx] for idx in indices])
        brier_v2_b = np.mean([v2_brier[idx] for idx in indices])
        brier_diffs.append(brier_v2_b - brier_v1_b)

    # p-value = fraction of bootstrap samples where the difference is 0 or opposite in sign to the observed difference
    # NDCG: we expect v2 > v1 (difference > 0)
    p_ndcg = np.mean(np.array(ndcg_diffs) <= 0) if observed_ndcg_diff > 0 else np.mean(np.array(ndcg_diffs) >= 0)
    # Brier Score: we expect v2 < v1 (difference < 0)
    p_brier = np.mean(np.array(brier_diffs) >= 0) if observed_brier_diff < 0 else np.mean(np.array(brier_diffs) <= 0)

    return {
        "ndcg_observed_difference": float(observed_ndcg_diff),
        "ndcg_p_value": float(p_ndcg),
        "brier_observed_difference": float(observed_brier_diff),
        "brier_p_value": float(p_brier)
    }


# =============================================================================
# Main Evaluation Orchestration
# =============================================================================

def print_markdown_table(v1_m: dict, v2_m: dict, sig_results: dict):
    """Generate and print a publication-quality comparison table in markdown."""
    print("\n" + "=" * 80)
    print("                    TRACE-RCE v1 vs v2 Head-to-Head Comparison")
    print("=" * 80)
    print("| Metric | TRACE-RCE v1 (Rule-Based) | TRACE-RCE v2 (Trainable) | Improvement | p-value |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    metrics_to_show = [
        ("Precision@1", "precision_at_1", "+", "higher"),
        ("Precision@3", "precision_at_3", "+", "higher"),
        ("Recall@3", "recall_at_3", "+", "higher"),
        ("MRR", "mrr", "+", "higher"),
        ("NDCG@3", "ndcg_at_3", "+", "higher"),
        ("Brier Score", "brier_score", "-", "lower"),
        ("Expected Calibration Error (ECE)", "expected_calibration_error", "-", "lower"),
    ]

    for label, key, sign, direction in metrics_to_show:
        val1 = v1_m.get(key, 0.0)
        val2 = v2_m.get(key, 0.0)
        diff = val2 - val1
        
        # Determine improvement percentage or raw difference
        imp_str = f"{diff:+.4f}"
        
        # Significance indicators
        p_str = "n/a"
        if key == "ndcg_at_3":
            p_val = sig_results["ndcg_p_value"]
            p_str = f"{p_val:.4f}" + (" *" if p_val < 0.05 else "")
        elif key == "brier_score":
            p_val = sig_results["brier_p_value"]
            p_str = f"{p_val:.4f}" + (" *" if p_val < 0.05 else "")

        print(f"| {label} | {val1:.4f} | {val2:.4f} | {imp_str} | {p_str} |")

    # Latency and Size
    print(f"| Mean Latency (ms) | {v1_m['mean']:.2f} | {v2_m['mean']:.2f} | {v2_m['mean']-v1_m['mean']:+.2f} | - |")
    print(f"| p95 Latency (ms) | {v1_m['p95']:.2f} | {v2_m['p95']:.2f} | {v2_m['p95']-v1_m['p95']:+.2f} | - |")
    print("=" * 80)
    print("(* indicates statistical significance at alpha=0.05)\n")


def main():
    # Setup paths
    config_path = "backend/research/trace_rce_v2/configs/default.yaml"
    checkpoint_path = "backend/research/trace_rce_v2/checkpoints/best_model.pt"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model checkpoint found at {checkpoint_path}.")
        print("Please run the training script first: python -m backend.research.trace_rce_v2.training.train")
        return

    # Load checkpoint and config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    # 1. Load splits
    all_scenarios = generate_evaluation_dataset()
    _, _, test_scenarios = stratified_split(
        all_scenarios,
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["training"]["seed"]
    )

    _, _, test_loader = create_dataloaders(
        augmentation_factor=0,
        batch_size=config["training"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["training"]["seed"]
    )

    # 2. Run TRACE-RCE v1 Evaluation
    v1_metrics, v1_raw = asyncio.run(run_v1_evaluation(test_scenarios))

    # 3. Load TRACE-RCE v2 Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    # 4. Run TRACE-RCE v2 Evaluation
    v2_metrics, v2_raw = run_v2_evaluation(model, test_loader, device)

    # 5. Run Significance Test
    sig_results = paired_bootstrap_test(v1_raw, v2_raw, n_resamples=config["evaluation"]["n_bootstrap"])

    # 6. Output report
    print_markdown_table(v1_metrics, v2_metrics, sig_results)

    # Save metrics comparison JSON
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    comparison_path = os.path.join(results_dir, "head_to_head_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "v1_metrics": v1_metrics,
            "v2_metrics": v2_metrics,
            "significance": sig_results,
            "model_size_params": model.count_parameters()
        }, f, indent=2)
    logger.info(f"Comparison metrics saved to {comparison_path}")


if __name__ == "__main__":
    main()
