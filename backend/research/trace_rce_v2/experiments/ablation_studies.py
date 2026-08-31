"""
TRACE-RCE v2 — Systematic Ablation Studies
============================================
Runs a suite of ablation experiments covering:
1. Embedding dimensions: 16, 32, 64
2. Attention heads: 1, 2, 4 (using d_model=32)
3. Ranking Loss functions: ListMLE, ListNet, RankNet
4. Modality ablation: Removing visual, env, severity, knowledge, or history one at a time.

For each experiment, a fresh model is trained to convergence on the train split
and evaluated on the held-out test split, saving comparative metrics.
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple

from backend.research.trace_rce_v2.training.train import DEFAULT_CONFIG, set_seed
from backend.research.trace_rce_v2.dataset.dataloader import create_dataloaders
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.models.trace_rce_v2 import TRACERCEv2
from backend.research.trace_rce_v2.training.losses import CombinedLoss, BrierLoss
from backend.research.trace_rce_v2.training.trainer import Trainer
from backend.research.trace_rce_v2.evaluation.evaluator import run_v2_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nova.research.trace_rce_v2.ablations")

# =============================================================================
# Alternative Losses Implementation
# =============================================================================

class ListNetLoss(nn.Module):
    """
    ListNet Loss.
    Computes Cross-Entropy between softmax of predicted scores and softmax of target relevance.
    """
    def __init__(self):
        super().__init__()

    def forward(self, scores: torch.Tensor, target_ranks: torch.Tensor) -> torch.Tensor:
        # Target ranks: primary=1, secondary=2, others=3.
        # We invert it to relevance scores: primary=3, secondary=2, others=1.
        relevance = 4.0 - target_ranks.float()
        p_target = F.softmax(relevance, dim=1)
        p_pred = F.log_softmax(scores, dim=1)
        return -torch.sum(p_target * p_pred, dim=1).mean()

class RankNetLoss(nn.Module):
    """
    Pairwise RankNet Loss (Vectorized).
    Optimizes probability of correctly ordering all pairs where rank_i < rank_j.
    """
    def __init__(self):
        super().__init__()

    def forward(self, scores: torch.Tensor, target_ranks: torch.Tensor) -> torch.Tensor:
        # Pairwise differences: scores[:, i] - scores[:, j]
        # Shape: (batch_size, n_causes, 1) - (batch_size, 1, n_causes) -> (batch_size, n_causes, n_causes)
        diffs = scores.unsqueeze(2) - scores.unsqueeze(1)
        
        # Preference mask: True if cause i has higher relevance (lower rank integer) than cause j
        # Shape: (batch_size, n_causes, n_causes)
        preferred = target_ranks.unsqueeze(2) < target_ranks.unsqueeze(1)
        
        # BCE with target = 1.0 is -log(sigmoid(diffs)) = softplus(-diffs)
        pairwise_loss = F.softplus(-diffs)
        
        # Mask loss and compute mean over active preference pairs
        total_loss = torch.sum(pairwise_loss * preferred.float())
        n_pairs = preferred.float().sum().clamp(min=1.0)
        
        return total_loss / n_pairs

class CombinedAblationLoss(nn.Module):
    """Combined wrapper supporting ListMLE, ListNet, and RankNet."""
    def __init__(self, loss_type: str = "ListMLE", lambda_cal: float = 0.3):
        super().__init__()
        self.brier = BrierLoss()
        self.lambda_cal = lambda_cal
        self.loss_type = loss_type
        
        if loss_type == "ListMLE":
            from backend.research.trace_rce_v2.training.losses import ListMLELoss
            self.ranking_loss = ListMLELoss()
        elif loss_type == "ListNet":
            self.ranking_loss = ListNetLoss()
        elif loss_type == "RankNet":
            self.ranking_loss = RankNetLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, scores: torch.Tensor, confidences: torch.Tensor, target_ranks: torch.Tensor, binary_labels: torch.Tensor):
        l_rank = self.ranking_loss(scores, target_ranks)
        l_cal = self.brier(confidences, binary_labels)
        total = l_rank + self.lambda_cal * l_cal
        return total, l_rank, l_cal

# =============================================================================
# Custom Ablation Trainer to support Modality Dropout
# =============================================================================

class AblationTrainer(Trainer):
    """
    Subclass of Trainer that allows zeroing out specific modalities
    to measure performance degradation (ablation study).
    """
    def __init__(self, *args, ablated_modality: str = None, experiment_name: str = "default", **kwargs):
        super().__init__(*args, **kwargs)
        self.ablated_modality = ablated_modality
        self.experiment_name = experiment_name

    def _save_checkpoint(self, filename, epoch, metrics):
        # We ignore filename and save to a unique file for this experiment
        unique_filename = f"best_{self.experiment_name}.pt"
        path = os.path.join(self.checkpoint_dir, unique_filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        import time
        for attempt in range(5):
            try:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "metrics": metrics,
                    "config": self.config
                }, path)
                logger.info(f"==> Saved checkpoint to {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to write checkpoint to {path} (attempt {attempt+1}/5): {e}")
                if attempt == 4:
                    raise e
                time.sleep(0.5)

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_cal_loss = 0.0
        
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            # Move inputs to device and zero out the ablated modality
            inputs = {}
            for k in ["visual", "env", "severity", "knowledge", "historical"]:
                t = batch[k].to(self.device)
                if k == self.ablated_modality:
                    inputs[k] = torch.zeros_like(t)
                else:
                    inputs[k] = t

            ranks = batch["cause_ranking"].to(self.device)
            labels = batch["binary_labels"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss, l_rank, l_cal = self.loss_fn(
                    outputs["scores"],
                    outputs["confidences"],
                    ranks,
                    labels
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
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
        self.model.eval()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_cal_loss = 0.0

        all_preds = []
        all_gts = []
        all_confs = []
        all_outcomes = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {}
                for k in ["visual", "env", "severity", "knowledge", "historical"]:
                    t = batch[k].to(self.device)
                    if k == self.ablated_modality:
                        inputs[k] = torch.zeros_like(t)
                    else:
                        inputs[k] = t

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

        n_samples = len(self.val_loader.dataset)
        val_loss = total_loss / n_samples
        
        from backend.research.root_cause_engine.metrics import compute_metrics_package
        metrics = compute_metrics_package(all_preds, all_gts, all_confs, all_outcomes)
        metrics["rank_loss"] = total_rank_loss / n_samples
        metrics["cal_loss"] = total_cal_loss / n_samples

        return val_loss, metrics


# =============================================================================
# Helper to Run a Single Ablation Experiment
# =============================================================================

def run_experiment(
    config: Dict[str, Any],
    experiment_name: str,
    ablated_modality: str = None
) -> Dict[str, float]:
    """Train and evaluate the model for a specific parameter configuration."""
    set_seed(config["training"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        augmentation_factor=config["data"]["augmentation_factor"],
        batch_size=config["training"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["training"]["seed"]
    )

    # Initialize model with current configs
    model = TRACERCEv2(
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_causes=config["model"]["n_causes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-5
    )

    loss_fn = CombinedAblationLoss(
        loss_type=config["training"].get("loss_type", "ListMLE"),
        lambda_cal=config["training"]["lambda_cal"]
    )

    # Reduce epochs/patience for faster ablation runs
    config_adjusted = copy_dict(config)
    config_adjusted["training"]["epochs"] = 60
    config_adjusted["training"]["patience"] = 10
    config_adjusted["paths"]["checkpoint_dir"] = os.path.join(
        config["paths"]["checkpoint_dir"], "ablations"
    )

    trainer = AblationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config_adjusted,
        device=device,
        ablated_modality=ablated_modality,
        experiment_name=experiment_name
    )

    # Train model
    trainer.train()

    # Load best checkpoint from trainer run
    best_ckpt_path = os.path.join(config_adjusted["paths"]["checkpoint_dir"], f"best_{experiment_name}.pt")
    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on test set (with corresponding modality zeroed out if ablated)
    # We call run_v2_evaluation but pass a wrapper loader that zeros out the modality
    class AblatedLoader:
        def __init__(self, loader, modality):
            self.loader = loader
            self.modality = modality
            self.dataset = loader.dataset

        def __iter__(self):
            for batch in self.loader:
                if self.modality:
                    batch[self.modality] = torch.zeros_like(batch[self.modality])
                yield batch

    ablated_test_loader = AblatedLoader(test_loader, ablated_modality)
    metrics, _ = run_v2_evaluation(model, ablated_test_loader, device)

    logger.info(f"Experiment {experiment_name} Complete: Test NDCG@3 = {metrics['ndcg_at_3']:.4f}")
    return metrics

def copy_dict(d: dict) -> dict:
    return json.loads(json.dumps(d))

# =============================================================================
# Main Orchestration Loop
# =============================================================================

def main():
    config = copy_dict(DEFAULT_CONFIG)
    config["paths"]["checkpoint_dir"] = "backend/research/trace_rce_v2/checkpoints"
    config["paths"]["results_dir"] = "backend/research/trace_rce_v2/experiments"
    
    results = {}

    # --- 1. Embedding Dimension Studies ---
    logger.info("Starting Embedding Dimension studies...")
    for d in [16, 32, 64]:
        name = f"d_model_{d}"
        run_cfg = copy_dict(config)
        run_cfg["model"]["d_model"] = d
        # Adjust n_heads to divide d
        run_cfg["model"]["n_heads"] = 2 if d != 16 else 1
        results[name] = run_experiment(run_cfg, name)

    # --- 2. Attention Heads Studies ---
    logger.info("Starting Attention Heads studies...")
    # d_model=32 is divisible by 1, 2, 4, 8
    for h in [1, 2, 4, 8]:
        name = f"n_heads_{h}"
        run_cfg = copy_dict(config)
        run_cfg["model"]["d_model"] = 32
        run_cfg["model"]["n_heads"] = h
        results[name] = run_experiment(run_cfg, name)

    # --- 3. Ranking Loss Studies ---
    logger.info("Starting Ranking Loss studies...")
    for loss in ["ListMLE", "ListNet", "RankNet"]:
        name = f"loss_{loss}"
        run_cfg = copy_dict(config)
        run_cfg["training"]["loss_type"] = loss
        results[name] = run_experiment(run_cfg, name)

    # --- 4. Modality Omission Studies ---
    logger.info("Starting Modality Omission studies...")
    MODALITY_ORDER = ["visual", "env", "severity", "knowledge", "historical"]
    for m in MODALITY_ORDER:
        name = f"remove_{m}"
        run_cfg = copy_dict(config)
        results[name] = run_experiment(run_cfg, name, ablated_modality=m)

    # Save all results to disk
    out_dir = config["paths"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*80)
    print("                      TRACE-RCE v2 Ablation Matrix")
    print("="*80)
    print("| Configuration | NDCG@3 | MRR | Precision@1 | Brier Score | ECE |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name, m in results.items():
        print(f"| {name:20s} | {m['ndcg_at_3']:.4f} | {m['mrr']:.4f} | {m['precision_at_1']:.4f} | {m.get('brier_score', 0.0):.4f} | {m.get('expected_calibration_error', 0.0):.4f} |")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
