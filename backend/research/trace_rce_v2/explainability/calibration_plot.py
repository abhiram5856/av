"""
TRACE-RCE v2 — Probability Calibration Plot
=============================================
Generates a reliability diagram (calibration curve) for both v1 and v2 models
on the test set of the NOVA-RCD dataset.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import asyncio
from typing import Dict, List, Tuple

# Add root directory to sys path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.root_cause_engine.dataset import EvalScenario
from backend.research.trace_rce_v2.dataset.novarcd_dataset import load_split_records, deserialize_context, create_novarcd_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.evaluation.evaluator import run_v1_evaluation, run_v2_evaluation
from backend.research.root_cause_engine.metrics import compute_calibration_errors

logger = logging.getLogger("nova.research.trace_rce_v2.calibration_plot")

def compute_calibration_curve(
    confidences: List[float],
    outcomes: List[int],
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidences = np.array(confidences)
    outcomes = np.array(outcomes)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if i == n_bins - 1:
            in_bin = in_bin | (confidences == bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        bin_sizes.append(prop_in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(outcomes[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(confidence_in_bin)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            
    return np.array(bin_accuracies), np.array(bin_confidences), np.array(bin_sizes)

def plot_reliability_diagrams(
    v1_confs: List[float], v1_outs: List[int], v1_ece: float,
    v2_confs: List[float], v2_outs: List[int], v2_ece: float,
    save_path: str,
    n_bins: int = 10
):
    v1_accs, v1_bin_confs, _ = compute_calibration_curve(v1_confs, v1_outs, n_bins)
    v2_accs, v2_bin_confs, _ = compute_calibration_curve(v2_confs, v2_outs, n_bins)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot V1 (Rule-Based)
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    v1_valid = ~np.isnan(v1_accs)
    ax1.plot(v1_bin_confs[v1_valid], v1_accs[v1_valid], "s-", color="red", label="TRACE-RCE v1")
    ax1.bar(
        np.linspace(0.05, 0.95, n_bins)[v1_valid],
        np.abs(v1_accs[v1_valid] - v1_bin_confs[v1_valid]),
        width=0.08, alpha=0.15, edgecolor="red", color="red", label="Calibration Gap"
    )
    ax1.set_title(f"TRACE-RCE v1 (Rule-Based)\nECE = {v1_ece:.4f}", fontsize=12)
    ax1.set_xlabel("Predicted Confidence")
    ax1.set_ylabel("Empirical Accuracy")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot V2 (Trainable)
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    v2_valid = ~np.isnan(v2_accs)
    ax2.plot(v2_bin_confs[v2_valid], v2_accs[v2_valid], "s-", color="green", label="TRACE-RCE v2")
    ax2.bar(
        np.linspace(0.05, 0.95, n_bins)[v2_valid],
        np.abs(v2_accs[v2_valid] - v2_bin_confs[v2_valid]),
        width=0.08, alpha=0.15, edgecolor="green", color="green", label="Calibration Gap"
    )
    ax2.set_title(f"TRACE-RCE v2 (Neural)\nECE = {v2_ece:.4f}", fontsize=12)
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Empirical Accuracy")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Calibration plot saved to {save_path}")

async def main():
    checkpoint_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints\best_novarcd_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model checkpoint found at {checkpoint_path}.")
        return

    # Load splits
    _, _, test_records = load_split_records()
    _, _, test_loader = create_novarcd_dataloaders(batch_size=32, augmentation_factor=0)

    # Run TRACE-RCE v1
    print("Running V1 Calibration Evaluation...")
    test_scenarios = [EvalScenario(context=deserialize_context(r["context"]), ground_truth_causes=r["ground_truth_causes"], case_id=r["case_id"], description="") for r in test_records]
    v1_metrics, v1_raw = await run_v1_evaluation(test_scenarios)
    v1_flat_confs = [c for sub in v1_raw["confidences"] for c in sub]
    v1_flat_outs = [o for sub in v1_raw["outcomes"] for o in sub]
    v1_ece, _ = compute_calibration_errors(v1_flat_confs, v1_flat_outs, n_bins=10)

    # Load TRACE-RCE v2 Model
    print("Running V2 Calibration Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    
    # Run TRACE-RCE v2
    v2_metrics, v2_raw = run_v2_evaluation(model, test_loader, device)
    v2_flat_confs = [c for sub in v2_raw["confidences"] for c in sub]
    v2_flat_outs = [o for sub in v2_raw["outcomes"] for o in sub]
    v2_ece, _ = compute_calibration_errors(v2_flat_confs, v2_flat_outs, n_bins=10)

    # Plot
    save_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\evaluation\calibration_curve.png"
    plot_reliability_diagrams(
        v1_flat_confs, v1_flat_outs, v1_ece,
        v2_flat_confs, v2_flat_outs, v2_ece,
        save_path,
        n_bins=10
    )

if __name__ == "__main__":
    asyncio.run(main())
