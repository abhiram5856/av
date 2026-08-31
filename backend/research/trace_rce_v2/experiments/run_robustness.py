"""
TRACE-RCE v2 — Robustness & Perturbation Testing
===================================================
Evaluates the trained TRACE-RCE v2 model under adverse conditions:
1. Missing Weather data (all env features set to 0.0)
2. Missing Knowledge RAG context (all knowledge features set to 0.0)
3. Low-Confidence Vision (conf=0.3, GradCAM=0.1)
4. Noisy Inputs (Gaussian noise with σ=0.20 added to all inputs)
5. Extreme Environment (humidity=1.0, temp=1.0, rainfall=1.0, wetness=1.0)
6. Out-of-Distribution (OOD) Cases (perturbed scenarios from dataset)

This script analyzes the degradation of v2 to assess its real-world
reliability when hardware or external services fail.
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, Any, List

from backend.research.root_cause_engine.dataset import generate_evaluation_dataset
from backend.research.trace_rce_v2.dataset.dataloader import stratified_split, create_dataloaders
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.evaluation.evaluator import run_v2_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nova.research.trace_rce_v2.robustness")

class PerturbedLoader:
    """Wrapper around DataLoader to dynamically apply input perturbations."""
    def __init__(self, loader, perturbation_type: str):
        self.loader = loader
        self.perturbation_type = perturbation_type
        self.dataset = loader.dataset

    def __iter__(self):
        for batch in self.loader:
            batch_perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if self.perturbation_type == "missing_weather":
                batch_perturbed["env"] = torch.zeros_like(batch_perturbed["env"])
                
            elif self.perturbation_type == "missing_knowledge":
                batch_perturbed["knowledge"] = torch.zeros_like(batch_perturbed["knowledge"])
                
            elif self.perturbation_type == "low_confidence_vision":
                batch_perturbed["visual"][:, 0] = 0.3  # conf
                batch_perturbed["visual"][:, 1] = 0.1  # coverage
                batch_perturbed["visual"][:, 2] = 0.1  # peak
                
            elif self.perturbation_type == "noisy_inputs":
                for modality in ["visual", "env", "severity", "knowledge", "historical"]:
                    noise = torch.randn_like(batch_perturbed[modality]) * 0.20
                    batch_perturbed[modality] = (batch_perturbed[modality] + noise).clamp(0.0, 1.0)
                    
            elif self.perturbation_type == "extreme_environment":
                batch_perturbed["env"][:, 0] = 1.0  # humidity
                batch_perturbed["env"][:, 1] = 1.0  # temperature
                batch_perturbed["env"][:, 2] = 1.0  # rainfall
                batch_perturbed["env"][:, 3] = 1.0  # leaf wetness
                
            yield batch_perturbed

def evaluate_ood_scenarios(model: torch.nn.Module, test_scenarios: List, device: torch.device) -> Dict[str, float]:
    """Evaluate specifically on out-of-distribution (ood_disease) test cases."""
    from backend.research.trace_rce_v2.dataset.dataset import RCEDataset
    from torch.utils.data import DataLoader
    from backend.research.trace_rce_v2.dataset.dataloader import rce_collate_fn
    
    # Filter scenarios tagged as 'ood_disease'
    ood_scenarios = [s for s in test_scenarios if s.metadata.get("perturbation") == "ood_disease"]
    if not ood_scenarios:
        # Fallback to contradictory if no ood cases in test split
        ood_scenarios = [s for s in test_scenarios if s.metadata.get("perturbation") in ("contradictory_evidence", "ood_disease")]

    ood_dataset = RCEDataset(ood_scenarios, augmentation_factor=0, seed=42)
    ood_loader = DataLoader(ood_dataset, batch_size=8, shuffle=False, collate_fn=rce_collate_fn)
    
    metrics, _ = run_v2_evaluation(model, ood_loader, device)
    return metrics

def main():
    checkpoint_path = "backend/research/trace_rce_v2/checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No trained model checkpoint found at {checkpoint_path}.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    
    # 1. Load test scenarios and dataloader
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

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    robustness_results = {}

    # --- 1. Baseline ---
    logger.info("Evaluating baseline test performance...")
    robustness_results["baseline"] = run_v2_evaluation(model, test_loader, device)[0]

    # --- 2. Missing Weather ---
    logger.info("Evaluating missing weather perturbation...")
    loader = PerturbedLoader(test_loader, "missing_weather")
    robustness_results["missing_weather"] = run_v2_evaluation(model, loader, device)[0]

    # --- 3. Missing Knowledge ---
    logger.info("Evaluating missing knowledge perturbation...")
    loader = PerturbedLoader(test_loader, "missing_knowledge")
    robustness_results["missing_knowledge"] = run_v2_evaluation(model, loader, device)[0]

    # --- 4. Low-Confidence Vision ---
    logger.info("Evaluating low-confidence vision perturbation...")
    loader = PerturbedLoader(test_loader, "low_confidence_vision")
    robustness_results["low_confidence_vision"] = run_v2_evaluation(model, loader, device)[0]

    # --- 5. Noisy Inputs ---
    logger.info("Evaluating noisy inputs perturbation...")
    loader = PerturbedLoader(test_loader, "noisy_inputs")
    robustness_results["noisy_inputs"] = run_v2_evaluation(model, loader, device)[0]

    # --- 6. Extreme Environment ---
    logger.info("Evaluating extreme environment perturbation...")
    loader = PerturbedLoader(test_loader, "extreme_environment")
    robustness_results["extreme_environment"] = run_v2_evaluation(model, loader, device)[0]

    # --- 7. OOD Scenarios ---
    logger.info("Evaluating OOD scenarios...")
    robustness_results["ood_scenarios"] = evaluate_ood_scenarios(model, test_scenarios, device)

    # Save results
    results_dir = config["paths"]["results_dir"]
    out_path = os.path.join(results_dir, "robustness_results.json")
    with open(out_path, "w") as f:
        json.dump(robustness_results, f, indent=2)

    print("\n" + "="*80)
    print("                      TRACE-RCE v2 Robustness Analysis")
    print("="*80)
    print("| Perturbation | NDCG@3 | MRR | Precision@1 | Delta NDCG@3 | Status |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    baseline_ndcg = robustness_results["baseline"]["ndcg_at_3"]
    for name, m in robustness_results.items():
        diff_ndcg = m["ndcg_at_3"] - baseline_ndcg
        # Status checks
        status = "PASSED" if diff_ndcg >= -0.15 else "DEGRADED"
        if name == "baseline":
            status = "BASELINE"
        print(f"| {name:25s} | {m['ndcg_at_3']:.4f} | {m['mrr']:.4f} | {m['precision_at_1']:.4f} | {diff_ndcg:+.4f} | {status} |")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
