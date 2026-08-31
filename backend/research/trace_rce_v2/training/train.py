"""
TRACE-RCE v2 — Training Entry Point
======================================
Sets up logging, configures random seeds for absolute reproducibility,
builds the dataloaders, initializes the model, optimizer, scheduler,
and loss, and runs the training process.

At the end of training, it saves:
1. The best model checkpoint (best_model.pt)
2. The full training history (history.json)
3. A plot of training curves (learning_curves.png)
"""

import os
import json
import random
import logging
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Try importing yaml, fall back to default dict if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from backend.research.trace_rce_v2.dataset.dataloader import create_dataloaders, print_dataset_statistics
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.training.losses import CombinedLoss
from backend.research.trace_rce_v2.training.trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("nova.research.trace_rce_v2.train")

DEFAULT_CONFIG = {
    "model": {
        "d_model": 32,
        "n_heads": 2,
        "n_causes": 13,
        "dropout": 0.1,
    },
    "training": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 300,
        "batch_size": 32,
        "patience": 30,
        "min_delta": 1e-4,
        "lambda_cal": 0.3,
        "grad_clip": 1.0,
        "seed": 42,
        "use_amp": False,
        "lr_scheduler": {
            "type": "cosine",
            "T_max": 150,
            "eta_min": 1e-5
        }
    },
    "data": {
        "augmentation_factor": 5,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "stratify_by": "disease_family"
    },
    "paths": {
        "checkpoint_dir": "backend/research/trace_rce_v2/checkpoints",
        "results_dir": "backend/research/trace_rce_v2/experiments"
    }
}

def load_config(config_path: str) -> dict:
    """Load configuration from YAML or fall back to default."""
    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                logger.info(f"Successfully loaded config from {config_path}")
                return cfg
        except Exception as e:
            logger.warning(f"Failed to load YAML config: {e}. Using defaults.")
    else:
        logger.info("Using default configuration.")
    return DEFAULT_CONFIG

def set_seed(seed: int):
    """Set all random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Deterministic seed set to {seed}")

def plot_learning_curves(history: dict, save_dir: str):
    """Generate and save training history plots."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue", alpha=0.7)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="red", alpha=0.7)
    ax1.set_title("Training vs Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Metric plot (NDCG@3 and MRR)
    ax2.plot(epochs, history["val_ndcg"], label="Val NDCG@3", color="green", alpha=0.7)
    ax2.plot(epochs, history["val_mrr"], label="Val MRR", color="orange", alpha=0.7)
    ax2.set_title("Validation Ranking Performance")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Learning curves saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Train TRACE-RCE v2 Multimodal Model")
    parser.add_argument(
        "--config",
        type=str,
        default="backend/research/trace_rce_v2/configs/default.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    # 1. Load config
    config = load_config(args.config)

    # 2. Set seed
    set_seed(config["training"]["seed"])

    # 3. Create loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        augmentation_factor=config["data"]["augmentation_factor"],
        batch_size=config["training"]["batch_size"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=config["training"]["seed"]
    )
    print_dataset_statistics(train_loader, val_loader, test_loader)

    # 4. Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # 5. Build model
    model = build_model(config["model"]).to(device)
    logger.info(model.parameter_summary())

    # 6. Setup optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    sched_cfg = config["training"]["lr_scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_cfg["T_max"],
        eta_min=sched_cfg["eta_min"]
    )

    # 7. Setup combined loss
    loss_fn = CombinedLoss(lambda_cal=config["training"]["lambda_cal"])

    # 8. Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=device
    )

    # 9. Train
    history = trainer.train()

    # 10. Save history and plots
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    history_path = os.path.join(results_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    plot_learning_curves(history, results_dir)

if __name__ == "__main__":
    main()
