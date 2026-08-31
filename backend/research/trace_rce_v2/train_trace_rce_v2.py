"""
NOVA Root Cause Analysis — TRACE-RCE v2 Custom Training
======================================================
Trains the multi-head self-attention TRACE-RCE v2 neural model
on the custom NOVA-RCD dataset using the optimized ListNet listwise ranking loss.

Checkpoint:
  - `backend/research/trace_rce_v2/checkpoints/best_novarcd_model.pt`
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from backend.research.trace_rce_v2.training.train import set_seed, DEFAULT_CONFIG
from backend.research.trace_rce_v2.dataset.novarcd_dataset import create_novarcd_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import TRACERCEv2
from backend.research.trace_rce_v2.training.losses import BrierLoss
from backend.research.trace_rce_v2.experiments.ablation_studies import ListNetLoss
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nova.research.trace_rce_v2.train_novarcd")

class CustomListNetLoss(nn.Module):
    """ListNet ranking loss + regularized Brier calibration loss."""
    def __init__(self, lambda_cal: float = 0.5):
        super().__init__()
        self.ranking = ListNetLoss()
        self.brier = BrierLoss()
        self.lambda_cal = lambda_cal

    def forward(self, scores: torch.Tensor, confs: torch.Tensor, ranks: torch.Tensor, labels: torch.Tensor):
        l_rank = self.ranking(scores, ranks)
        l_cal = self.brier(confs, labels)
        total = l_rank + self.lambda_cal * l_cal
        return total, l_rank, l_cal

def train_novarcd():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training TRACE-RCE v2 on device: {device}")
    
    # 1. Dataloaders
    train_loader, val_loader, test_loader = create_novarcd_dataloaders(
        batch_size=32,
        augmentation_factor=2,
        seed=42
    )
    
    # 2. Build model
    config = DEFAULT_CONFIG.copy()
    config["model"]["d_model"] = 32
    config["model"]["n_heads"] = 8  # Best head config from ablation studies
    config["model"]["dropout"] = 0.1
    
    model = TRACERCEv2(
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_causes=config["model"]["n_causes"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    loss_fn = CustomListNetLoss(lambda_cal=0.5)
    
    # Early stopping config
    best_val_ndcg = 0.0
    patience = 20
    patience_counter = 0
    epochs = 80
    
    checkpoint_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_novarcd_model.pt")
    
    history = {"train_loss": [], "val_loss": [], "val_ndcg": [], "val_ece": []}
    
    logger.info(f"Starting custom training loop for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                "visual": batch["visual"].to(device),
                "env": batch["env"].to(device),
                "severity": batch["severity"].to(device),
                "knowledge": batch["knowledge"].to(device),
                "historical": batch["historical"].to(device)
            }
            ranks = batch["cause_ranking"].to(device)
            labels = batch["binary_labels"].to(device)
            
            outputs = model(inputs)
            loss, _, _ = loss_fn(outputs["scores"], outputs["confidences"], ranks, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_total += loss.item() * ranks.size(0)
            
        scheduler.step()
        train_loss_mean = train_loss_total / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss_total = 0.0
        
        all_preds = []
        all_gts = []
        all_confs = []
        all_outcomes = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    "visual": batch["visual"].to(device),
                    "env": batch["env"].to(device),
                    "severity": batch["severity"].to(device),
                    "knowledge": batch["knowledge"].to(device),
                    "historical": batch["historical"].to(device)
                }
                ranks = batch["cause_ranking"].to(device)
                labels = batch["binary_labels"].to(device)
                
                outputs = model(inputs)
                loss, _, _ = loss_fn(outputs["scores"], outputs["confidences"], ranks, labels)
                val_loss_total += loss.item() * ranks.size(0)
                
                batch_scores = outputs["scores"].cpu()
                batch_confs = outputs["confidences"].cpu()
                batch_ranks = batch["cause_ranking"]
                
                for b in range(batch_scores.size(0)):
                    sorted_indices = torch.argsort(batch_scores[b], descending=True)
                    pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                    all_preds.append(pred_cause_ids)
                    
                    gt_indices = (batch_ranks[b] <= 2).nonzero(as_tuple=True)[0]
                    gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                    all_gts.append(gt_cause_ids)
                    
                    confs_b = []
                    outcomes_b = []
                    for idx in sorted_indices:
                        confs_b.append(batch_confs[b, idx].item())
                        outcomes_b.append(1 if CAUSE_IDS[idx.item()] in gt_cause_ids else 0)
                    all_confs.append(confs_b)
                    all_outcomes.append(outcomes_b)
                    
        val_loss_mean = val_loss_total / len(val_loader.dataset)
        
        from backend.research.root_cause_engine.metrics import compute_metrics_package
        metrics = compute_metrics_package(all_preds, all_gts, all_confs, all_outcomes)
        val_ndcg = metrics["ndcg_at_3"]
        val_ece = metrics.get("expected_calibration_error", 0.0)
        
        history["train_loss"].append(train_loss_mean)
        history["val_loss"].append(val_loss_mean)
        history["val_ndcg"].append(val_ndcg)
        history["val_ece"].append(val_ece)
        
        logger.info(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss_mean:.4f} | Val Loss: {val_loss_mean:.4f} | Val NDCG@3: {val_ndcg:.4f} | Val ECE: {val_ece:.4f}")
        
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "metrics": metrics,
                "history": history
            }, best_checkpoint_path)
            logger.info(f"==> Saved new best model checkpoint (Val NDCG@3: {val_ndcg:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # Save history to results
    results_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\experiments"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "novarcd_history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    logger.info("Training Completed!")

if __name__ == "__main__":
    train_novarcd()
