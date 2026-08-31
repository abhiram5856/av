"""
TRACE-RCE v2 — Cross-Modal Attention Visualization
======================================================
Extracts and visualizes the modality-level cross-attention weights.

Attention Matrix Interpretation
-------------------------------
The attention output is a 5x5 heatmap.
Let the rows be the query modality (Q) and the columns be the key/value
modality (K/V).

A entry Heatmap[i, j] represents how much modality i "attended to"
modality j to update its own context representation.
This shows which modalities the model learns are linked (e.g., visual
referencing environmental context to identify weather-related fungal spread).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict

from backend.research.trace_rce_v2.dataset.dataloader import create_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model

logger = logging.getLogger("nova.research.trace_rce_v2.attention_viz")

MODALITIES = ["Visual", "Environmental", "Severity", "Knowledge", "Historical"]

def compute_average_attention(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> np.ndarray:
    """Run model on dataloader and return averaged 5x5 attention weights matrix."""
    model.eval()
    accum_attn = np.zeros((5, 5))
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                k: batch[k].to(device)
                for k in ["visual", "env", "severity", "knowledge", "historical"]
            }
            outputs = model(inputs, return_attention=True)
            attn = outputs["attn_weights"].cpu().numpy()  # (batch_size, 5, 5)

            accum_attn += attn.sum(axis=0)
            total_samples += attn.shape[0]

    return accum_attn / total_samples

def plot_attention_heatmap(attn_matrix: np.ndarray, save_path: str):
    """Plot a 5x5 heatmap with annotations and labels."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(attn_matrix, cmap="YlGnBu", vmin=0.0, vmax=0.5)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Score", rotation=-90, va="bottom")

    # Set tick labels
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels(MODALITIES, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(MODALITIES)

    # Annotate entries with values
    for i in range(5):
        for j in range(5):
            val = attn_matrix[i, j]
            # Choose color depending on threshold for readability
            color = "white" if val > 0.25 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight="bold")

    ax.set_title("Cross-Modal Self-Attention Matrix (Average)", fontsize=14, pad=15)
    ax.set_xlabel("Key/Value Modality (K/V)", fontsize=11, labelpad=10)
    ax.set_ylabel("Query Modality (Q)", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Attention matrix heatmap saved to {save_path}")

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

    # Compute averaged attention
    avg_attn = compute_average_attention(model, test_loader, device)

    # Save heatmap
    results_dir = config["paths"]["results_dir"]
    save_path = os.path.join(results_dir, "attention_matrix.png")
    plot_attention_heatmap(avg_attn, save_path)

if __name__ == "__main__":
    main()
