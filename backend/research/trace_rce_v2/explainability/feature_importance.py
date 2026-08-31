"""
TRACE-RCE v2 — Feature Importance (Integrated Gradients)
============================================================
Implements Integrated Gradients (IG) to calculate feature attribution scores
for all 19 raw input features.

Mathematical Formulation
------------------------
For a given input feature vector x and baseline x' (all zeros), the attribution
for feature i with respect to output score s_c for cause c is:

    IG_i(x) = (x_i - x'_i) × ∫_{0}^{1} ∂s_c(x' + α(x - x')) / ∂x_i dα

We approximate this integral using Riemann summation with M=50 steps:

    IG_i(x) ≈ (x_i - x'_i) × (1 / M) × Σ_{m=1}^{M} ∂s_c(x' + (m / M)(x - x')) / ∂x_i

Integrated Gradients satisfies the axiom of "Completeness" (the sum of attributions
equals the difference between output at input and output at baseline), making it
a rigorous, mathematically sound explainability technique for neural networks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple

from backend.research.trace_rce_v2.dataset.dataloader import create_dataloaders
from backend.research.trace_rce_v2.dataset.feature_extractor import FEATURE_NAMES, N_TOTAL_FEATURES
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model

logger = logging.getLogger("nova.research.trace_rce_v2.feature_importance")

# Flatten feature names for plotting
FLATTENED_FEATURE_NAMES = []
for modality in ["visual", "env", "severity", "knowledge", "historical"]:
    FLATTENED_FEATURE_NAMES.extend([f"{modality}_{name}" for name in FEATURE_NAMES[modality]])

def compute_integrated_gradients(
    model: torch.nn.Module,
    batch: dict,
    sample_idx: int,
    target_cause_idx: int,
    steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients for a single sample with respect to a target cause.
    """
    model.eval()

    # Extract single sample inputs and enable gradients
    inputs = {}
    baselines = {}
    for k in ["visual", "env", "severity", "knowledge", "historical"]:
        inputs[k] = batch[k][sample_idx:sample_idx+1].clone().detach().requires_grad_(True)
        # Baseline is a zero tensor of the same shape
        baselines[k] = torch.zeros_like(inputs[k])

    # Accumulate gradients at interpolated steps
    grad_sums = {k: torch.zeros_like(inputs[k]) for k in inputs}

    for step in range(steps + 1):
        alpha = step / steps
        interpolated = {}
        
        # Linear interpolation: baseline + alpha * (input - baseline)
        for k in inputs:
            interpolated[k] = baselines[k] + alpha * (inputs[k] - baselines[k])
            interpolated[k] = interpolated[k].clone().detach().requires_grad_(True)

        # Forward pass on interpolated inputs
        # Enable gradient computation temporarily
        with torch.enable_grad():
            outputs = model(interpolated, return_attention=False)
            target_score = outputs["scores"][0, target_cause_idx]

        # Backward pass to get gradients of target score w.r.t interpolated features
        grads = torch.autograd.grad(target_score, list(interpolated.values()))
        
        # Accumulate
        for idx, k in enumerate(inputs):
            grad_sums[k] += grads[idx]

    # Compute final IG = (input - baseline) * mean_gradient
    ig_attributions = []
    for k in inputs:
        mean_grad = grad_sums[k] / (steps + 1)
        ig = (inputs[k] - baselines[k]) * mean_grad
        ig_attributions.append(ig.detach().cpu().numpy()[0])

    # Flatten and return: length 19
    return np.concatenate(ig_attributions)

def compute_average_feature_importance(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    steps: int = 50
) -> np.ndarray:
    """Compute average absolute IG feature attribution across the dataset."""
    total_ig = np.zeros(N_TOTAL_FEATURES)
    count = 0

    for batch in dataloader:
        batch_size = batch["visual"].size(0)
        
        # Transfer batch to device
        device_batch = {
            k: batch[k].to(device)
            for k in ["visual", "env", "severity", "knowledge", "historical"]
        }

        # Run forward pass to find predicted top cause for each sample
        model.eval()
        with torch.no_grad():
            outputs = model(device_batch, return_attention=False)
            scores = outputs["scores"].cpu() # (batch_size, n_causes)

        # Compute IG for each sample w.r.t its top predicted cause
        for b in range(batch_size):
            top_cause_idx = torch.argmax(scores[b]).item()
            
            # Integrated Gradients needs to run on device
            ig = compute_integrated_gradients(
                model=model,
                batch=device_batch,
                sample_idx=b,
                target_cause_idx=top_cause_idx,
                steps=steps
            )
            total_ig += np.abs(ig) # Absolute attribution represents overall importance
            count += 1

    return total_ig / count

def plot_feature_importance(importances: np.ndarray, save_path: str):
    """Plot horizontal bar chart of feature importances."""
    # Sort features by importance
    sorted_indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 8))
    
    # Choose color palette based on modality prefix
    colors = []
    for idx in sorted_indices:
        name = FLATTENED_FEATURE_NAMES[idx]
        if "visual" in name:
            colors.append("#3b82f6")  # blue
        elif "env" in name:
            colors.append("#10b981")  # green
        elif "severity" in name:
            colors.append("#ef4444")  # red
        elif "knowledge" in name:
            colors.append("#a855f7")  # purple
        else:
            colors.append("#f59e0b")  # orange

    plt.barh(
        np.arange(N_TOTAL_FEATURES),
        importances[sorted_indices],
        color=colors,
        alpha=0.8,
        edgecolor="grey"
    )
    
    plt.yticks(np.arange(N_TOTAL_FEATURES), [FLATTENED_FEATURE_NAMES[idx] for idx in sorted_indices])
    plt.xlabel("Average Absolute Integrated Gradients Attribution", fontsize=11, labelpad=10)
    plt.title("TRACE-RCE v2 Feature Importance Breakdown", fontsize=13, pad=15)
    
    # Draw legend placeholders
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3b82f6", label="Visual Modality"),
        Patch(facecolor="#10b981", label="Environmental Modality"),
        Patch(facecolor="#ef4444", label="Severity Modality"),
        Patch(facecolor="#a855f7", label="Knowledge Modality"),
        Patch(facecolor="#f59e0b", label="Historical Modality"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")

def main():
    checkpoint_path = "backend/research/trace_rce_v2/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model checkpoint found at {checkpoint_path}.")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    # Load test split
    _, _, test_loader = create_dataloaders(
        augmentation_factor=0,
        batch_size=config["training"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["training"]["seed"]
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Compute averaged importance (steps=20 for faster test computation)
    logger.info("Computing Integrated Gradients on test split (M=20 steps per sample)...")
    importances = compute_average_feature_importance(model, test_loader, device, steps=20)

    # Save chart
    results_dir = config["paths"]["results_dir"]
    save_path = os.path.join(results_dir, "feature_importance.png")
    plot_feature_importance(importances, save_path)

if __name__ == "__main__":
    main()
