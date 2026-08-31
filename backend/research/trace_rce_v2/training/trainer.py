"""
TRACE-RCE v2 — Trainer
=========================
Implements the training and validation loops, early stopping, learning rate
scheduling, checkpointing, and metric tracking.

Reuses the metrics package from TRACE-RCE v1 (metrics.py) to ensure
strict comparability between versions.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.root_cause_engine.metrics import compute_metrics_package

logger = logging.getLogger("nova.research.trace_rce_v2.trainer")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Any,
        loss_fn: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        self.epochs = config["training"]["epochs"]
        self.patience = config["training"]["patience"]
        self.min_delta = config["training"]["min_delta"]
        self.checkpoint_dir = config["paths"]["checkpoint_dir"]
        self.use_amp = config["training"].get("use_amp", False)
        
        # Mixed precisionscaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Metric tracking
        self.best_val_ndcg = -1.0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_rank_loss": [],
            "train_cal_loss": [],
            "val_loss": [],
            "val_rank_loss": [],
            "val_cal_loss": [],
            "val_ndcg": [],
            "val_mrr": [],
            "val_p1": [],
            "val_ece": [],
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop with early stopping."""
        logger.info("Starting training for %d epochs...", self.epochs)
        
        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_epoch()
            val_loss, val_metrics = self._evaluate()

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_rank_loss"].append(train_metrics["rank_loss"])
            self.history["train_cal_loss"].append(train_metrics["cal_loss"])
            
            self.history["val_loss"].append(val_loss)
            self.history["val_rank_loss"].append(val_metrics["rank_loss"])
            self.history["val_cal_loss"].append(val_metrics["cal_loss"])
            
            self.history["val_ndcg"].append(val_metrics["ndcg_at_3"])
            self.history["val_mrr"].append(val_metrics["mrr"])
            self.history["val_p1"].append(val_metrics["precision_at_1"])
            self.history["val_ece"].append(val_metrics.get("expected_calibration_error", 0.0))

            logger.info(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val NDCG@3: {val_metrics['ndcg_at_3']:.4f} | "
                f"Val ECE: {val_metrics.get('expected_calibration_error', 0.0):.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Learning rate scheduling step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping check based on NDCG@3 (higher is better)
            val_ndcg = val_metrics["ndcg_at_3"]
            if val_ndcg > self.best_val_ndcg + self.min_delta:
                self.best_val_ndcg = val_ndcg
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt", epoch, val_metrics)
                logger.info(f"==> Saved new best model checkpoint (Val NDCG@3: {val_ndcg:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs of no improvement.")
                    break

        return self.history

    def _train_epoch(self) -> Dict[str, float]:
        """Perform one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_cal_loss = 0.0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Move inputs to device
            inputs = {
                k: batch[k].to(self.device)
                for k in ["visual", "env", "severity", "knowledge", "historical"]
            }
            ranks = batch["cause_ranking"].to(self.device)
            labels = batch["binary_labels"].to(self.device)

            # Forward pass under AMP
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss, l_rank, l_cal = self.loss_fn(
                    outputs["scores"],
                    outputs["confidences"],
                    ranks,
                    labels
                )

            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config["training"]["grad_clip"]
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * ranks.size(0)
            total_rank_loss += l_rank.item() * ranks.size(0)
            total_cal_loss += l_cal.item() * ranks.size(0)

        n_samples = len(self.train_loader.dataset)
        return {
            "loss": total_loss / n_samples,
            "rank_loss": total_rank_loss / n_samples,
            "cal_loss": total_cal_loss / n_samples,
        }

    def _evaluate(self) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the model on validation set."""
        self.model.eval()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_cal_loss = 0.0

        all_preds: List[List[str]] = []
        all_gts: List[List[str]] = []
        all_confs: List[List[float]] = []
        all_outcomes: List[List[int]] = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {
                    k: batch[k].to(self.device)
                    for k in ["visual", "env", "severity", "knowledge", "historical"]
                }
                ranks = batch["cause_ranking"].to(self.device)
                labels = batch["binary_labels"].to(self.device)

                outputs = self.model(inputs)
                loss, l_rank, l_cal = self.loss_fn(
                    outputs["scores"],
                    outputs["confidences"],
                    ranks,
                    labels
                )

                total_loss += loss.item() * ranks.size(0)
                total_rank_loss += l_rank.item() * ranks.size(0)
                total_cal_loss += l_cal.item() * ranks.size(0)

                # Process predictions for metric calculation
                batch_scores = outputs["scores"].cpu()
                batch_confs = outputs["confidences"].cpu()
                batch_ranks = batch["cause_ranking"]

                for b in range(batch_scores.size(0)):
                    # Predicted causes (sorted descending by score)
                    sorted_indices = torch.argsort(batch_scores[b], descending=True)
                    pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                    all_preds.append(pred_cause_ids)

                    # Ground truth causes: those with ranks 1 and 2
                    gt_indices = (batch_ranks[b] <= 2).nonzero(as_tuple=True)[0]
                    gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                    all_gts.append(gt_cause_ids)

                    # Confidences and binary outcomes for calibration calculation
                    # outcomes: 1 if prediction matches a ground truth, else 0
                    confs_b = []
                    outcomes_b = []
                    for idx in sorted_indices:
                        confs_b.append(batch_confs[b, idx].item())
                        outcomes_b.append(1 if CAUSE_IDS[idx.item()] in gt_cause_ids else 0)
                    
                    all_confs.append(confs_b)
                    all_outcomes.append(outcomes_b)

        n_samples = len(self.val_loader.dataset)
        val_loss = total_loss / n_samples

        # Compute ranking and calibration metrics
        metrics = compute_metrics_package(
            all_predictions=all_preds,
            all_ground_truths=all_gts,
            all_confidences=all_confs,
            all_outcomes=all_outcomes
        )

        metrics["rank_loss"] = total_rank_loss / n_samples
        metrics["cal_loss"] = total_cal_loss / n_samples

        return val_loss, metrics

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, Any]):
        """Save a model checkpoint to disk."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_ndcg": self.best_val_ndcg,
            "metrics": metrics,
            "config": self.config
        }, path)
