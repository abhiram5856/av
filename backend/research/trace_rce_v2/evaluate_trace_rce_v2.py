"""
NOVA Root Cause Analysis — TRACE-RCE v2 Evaluation Script
=========================================================
Evaluates the trained TRACE-RCE v2 model on the NOVA-RCD held-out test split,
prints all core metrics, and generates explainability visualizations
(learning curves, attention maps, feature importance, calibration reliability,
and primary cause confusion matrix).
"""

import os
import json
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Any

from backend.research.trace_rce_v2.dataset.novarcd_dataset import create_novarcd_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS, CAUSE_TO_IDX
from backend.research.root_cause_engine.metrics import compute_calibration_errors
from backend.research.trace_rce_v2.explainability.attention_viz import compute_average_attention
from backend.research.trace_rce_v2.explainability.feature_importance import compute_average_feature_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nova.research.trace_rce_v2.evaluate_novarcd")

ROOT_DIR = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI"
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "backend", "research", "trace_rce_v2", "checkpoints", "best_novarcd_model.pt")
RESULTS_DIR = os.path.join(ROOT_DIR, "backend", "research", "trace_rce_v2", "experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_learning_curves(history_file: str):
    """Plot training and validation loss & ECE curves."""
    with open(history_file, "r") as f:
        history = json.load(f)
        
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4")
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves (Loss)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_ndcg"], label="Val NDCG@3", color="#2ca02c")
    plt.plot(epochs, history["val_ece"], label="Val ECE", color="#d62728")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Metrics Tracking")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    out_path = os.path.join(RESULTS_DIR, "novarcd_learning_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plotted learning curves to {out_path}")

def plot_confusion_matrix(y_true: List[str], y_pred: List[str]):
    """Plot confusion matrix of primary causes."""
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    clean_labels = [l.replace("_", " ").title() for l in labels]
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Primary Root Cause Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, clean_labels, rotation=45, ha="right")
    plt.yticks(tick_marks, clean_labels)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel("Actual Primary Cause")
    plt.xlabel("Predicted Primary Cause")
    
    out_path = os.path.join(RESULTS_DIR, "novarcd_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plotted confusion matrix to {out_path}")

def plot_attention_matrix(avg_attn: np.ndarray):
    """Plot 5x5 cross-attention matrix heatmap."""
    modalities = ["Visual", "Env", "Severity", "Knowledge", "Historical"]
    
    plt.figure(figsize=(6, 5))
    plt.imshow(avg_attn, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("Average Cross-Modality Attention Weights")
    plt.colorbar()
    
    tick_marks = np.arange(len(modalities))
    plt.xticks(tick_marks, modalities)
    plt.yticks(tick_marks, modalities)
    
    thresh = avg_attn.max() / 2.
    for i in range(avg_attn.shape[0]):
        for j in range(avg_attn.shape[1]):
            plt.text(j, i, format(avg_attn[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if avg_attn[i, j] > thresh else "black")
                     
    plt.ylabel("Query Modality")
    plt.xlabel("Key Modality")
    
    out_path = os.path.join(RESULTS_DIR, "novarcd_attention_matrix.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plotted attention matrix to {out_path}")

def plot_feature_importance(importances: np.ndarray):
    """Plot Integrated Gradients feature importance bar chart."""
    # Master list of all 19 features extracted
    feature_names = [
        "vis_conf", "vis_cov", "vis_peak", "vis_entropy",
        "env_humidity", "env_temp", "env_rain", "env_wetness",
        "sev_base", "sev_env", "sev_soil", "sev_final",
        "rag_chunks", "rag_len", "rag_sources", "rag_keywords",
        "hist_total", "hist_frequent", "hist_recency"
    ]
    
    # Sort
    indices = np.argsort(importances)
    
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], color="#2ca02c", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Mean Absolute Attribution (IG)")
    plt.title("Feature Attribution w.r.t Predicted Root Cause")
    
    out_path = os.path.join(RESULTS_DIR, "novarcd_feature_importance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plotted feature importance to {out_path}")

def plot_calibration_curve(confs: List[float], outcomes: List[int]):
    """Plot reliability curve showing binned confidence vs accuracy."""
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_confs = []
    bin_accs = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        
        mask = [(c >= bin_lower) and (c < bin_upper) for c in confs]
        # handle last bin edge inclusion
        if i == n_bins - 1:
            mask = [(c >= bin_lower) and (c <= bin_upper) for c in confs]
            
        indices = [idx for idx, val in enumerate(mask) if val]
        
        if len(indices) > 0:
            bin_confs.append(np.mean([confs[idx] for idx in indices]))
            bin_accs.append(np.mean([outcomes[idx] for idx in indices]))
            bin_counts.append(len(indices))
            
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, "o-", color="#1f77b4", label="TRACE-RCE v2 (ListNet)")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Calibration Reliability Diagram")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    out_path = os.path.join(RESULTS_DIR, "novarcd_calibration_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plotted calibration curve to {out_path}")

def evaluate_novarcd():
    print(f"Loading custom model checkpoint from {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Trained checkpoint not found at {CHECKPOINT_PATH}. Run training first.")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = checkpoint["config"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    _, _, test_loader = create_novarcd_dataloaders(
        batch_size=config["training"].get("batch_size", 32),
        augmentation_factor=0,
        seed=42
    )
    
    all_preds = []
    all_gts = []
    all_confs = []
    all_outcomes = []
    
    y_true_primary = []
    y_pred_primary = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "visual": batch["visual"].to(device),
                "env": batch["env"].to(device),
                "severity": batch["severity"].to(device),
                "knowledge": batch["knowledge"].to(device),
                "historical": batch["historical"].to(device)
            }
            ranks = batch["cause_ranking"]
            
            outputs = model(inputs)
            
            batch_scores = outputs["scores"].cpu()
            batch_confs = outputs["confidences"].cpu()
            
            for b in range(batch_scores.size(0)):
                sorted_indices = torch.argsort(batch_scores[b], descending=True)
                pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                all_preds.append(pred_cause_ids)
                
                gt_indices = (ranks[b] <= 2).nonzero(as_tuple=True)[0]
                gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                all_gts.append(gt_cause_ids)
                
                # Primary cause tracker (only if not healthy, i.e., gt_cause_ids is not empty)
                if len(gt_cause_ids) > 0:
                    y_true_primary.append(gt_cause_ids[0])
                    # predicted primary
                    y_pred_primary.append(pred_cause_ids[0])
                
                confs_b = []
                outcomes_b = []
                for idx in sorted_indices:
                    confs_b.append(batch_confs[b, idx].item())
                    outcomes_b.append(1 if CAUSE_IDS[idx.item()] in gt_cause_ids else 0)
                all_confs.append(confs_b)
                all_outcomes.append(outcomes_b)
                
    # Flat arrays for calibration error
    flat_confs = [c for sub in all_confs for c in sub]
    flat_outcomes = [o for sub in all_outcomes for o in sub]
    
    from backend.research.root_cause_engine.metrics import compute_metrics_package
    metrics = compute_metrics_package(all_preds, all_gts, all_confs, all_outcomes)
    
    ece, _ = compute_calibration_errors(flat_confs, flat_outcomes, n_bins=10)
    brier = metrics["brier_score"]
    
    print("\n" + "="*50)
    print("        NOVA-RCD Root Cause Model Evaluation")
    print("="*50)
    print(f"Precision@1 : {metrics['precision_at_1']:.4f}")
    print(f"Precision@3 : {metrics['precision_at_3']:.4f}")
    print(f"Recall@3    : {metrics['recall_at_3']:.4f}")
    print(f"MRR         : {metrics['mrr']:.4f}")
    print(f"NDCG@3      : {metrics['ndcg_at_3']:.4f}")
    print(f"Brier Score : {brier:.4f}")
    print(f"ECE         : {ece:.4f}")
    print("="*50 + "\n")
    
    # Generate explainability figures
    history_file = os.path.join(RESULTS_DIR, "novarcd_history.json")
    if os.path.exists(history_file):
        plot_learning_curves(history_file)
        
    avg_attn = compute_average_attention(model, test_loader, device)
    plot_attention_matrix(avg_attn)
    
    importances = compute_average_feature_importance(model, test_loader, device, steps=20)
    plot_feature_importance(importances)
    
    plot_calibration_curve(flat_confs, flat_outcomes)
    
    if len(y_true_primary) > 0:
        plot_confusion_matrix(y_true_primary, y_pred_primary)
        
    print("Evaluation Complete!")

if __name__ == "__main__":
    evaluate_novarcd()
