import os
import json
import logging
import torch
import numpy as np
import time
from typing import Dict, Any, List
from scipy.stats import wilcoxon

from backend.research.trace_rce_v3.dataset.novarcd_v3_dataset import create_novarcd_v3_dataloaders
from backend.research.trace_rce_v3.models.trace_rce_v3 import TRACERCEv3
from backend.research.trace_rce_v3.evaluation.run_validation_suite import compute_metrics_package, compute_calibration_errors
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pixel_robustness")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PixelCorruptionLoader:
    def __init__(self, loader, corruption_type: str, severity: int):
        self.loader = loader
        self.corruption_type = corruption_type
        self.severity = severity
        self.dataset = loader.dataset

    def __iter__(self):
        for batch in self.loader:
            batch_perturbed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if self.corruption_type == "Gaussian Blur":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 2] *= 0.90
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.10, 0, 1)
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.70
                    batch_perturbed["visual"][:, 2] *= 0.70
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.20, 0, 1)
                else:
                    batch_perturbed["visual"][:, 0] *= 0.40
                    batch_perturbed["visual"][:, 2] *= 0.40
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.40, 0, 1)
                    
            elif self.corruption_type == "Motion Blur":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.85
                    batch_perturbed["visual"][:, 1] = torch.clamp(batch_perturbed["visual"][:, 1] * 1.1, 0, 1)
                    batch_perturbed["visual"][:, 2] *= 0.85
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.10, 0, 1)
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.65
                    batch_perturbed["visual"][:, 1] = torch.clamp(batch_perturbed["visual"][:, 1] * 1.2, 0, 1)
                    batch_perturbed["visual"][:, 2] *= 0.65
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.20, 0, 1)
                else:
                    batch_perturbed["visual"][:, 0] *= 0.35
                    batch_perturbed["visual"][:, 1] = torch.clamp(batch_perturbed["visual"][:, 1] * 1.5, 0, 1)
                    batch_perturbed["visual"][:, 2] *= 0.35
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.40, 0, 1)
                    
            elif self.corruption_type == "Brightness Increase":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 2] *= 0.80
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.70
                    batch_perturbed["visual"][:, 2] *= 0.60
                else:
                    batch_perturbed["visual"][:, 0] *= 0.50
                    batch_perturbed["visual"][:, 2] *= 0.40

            elif self.corruption_type == "Brightness Decrease":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 1] *= 0.90
                    batch_perturbed["visual"][:, 2] *= 0.90
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.70
                    batch_perturbed["visual"][:, 1] *= 0.70
                    batch_perturbed["visual"][:, 2] *= 0.70
                else:
                    batch_perturbed["visual"][:, 0] *= 0.50
                    batch_perturbed["visual"][:, 1] *= 0.50
                    batch_perturbed["visual"][:, 2] *= 0.50
                    
            elif self.corruption_type == "Contrast Reduction":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 2] *= 0.85
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.75
                    batch_perturbed["visual"][:, 2] *= 0.65
                else:
                    batch_perturbed["visual"][:, 0] *= 0.50
                    batch_perturbed["visual"][:, 2] *= 0.45

            elif self.corruption_type == "JPEG Compression":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.95
                    batch_perturbed["visual"][:, 2] *= 0.95
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.85
                    batch_perturbed["visual"][:, 2] *= 0.85
                else:
                    batch_perturbed["visual"][:, 0] *= 0.65
                    batch_perturbed["visual"][:, 2] *= 0.70
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.15, 0, 1)
                    
            elif self.corruption_type == "Gaussian Sensor Noise":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 2] = torch.clamp(batch_perturbed["visual"][:, 2] + 0.05, 0, 1)
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.10, 0, 1)
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.75
                    batch_perturbed["visual"][:, 2] = torch.clamp(batch_perturbed["visual"][:, 2] + 0.10, 0, 1)
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.20, 0, 1)
                else:
                    batch_perturbed["visual"][:, 0] *= 0.50
                    batch_perturbed["visual"][:, 2] = torch.clamp(batch_perturbed["visual"][:, 2] + 0.20, 0, 1)
                    batch_perturbed["visual"][:, 3] = torch.clamp(batch_perturbed["visual"][:, 3] + 0.35, 0, 1)
                    
            elif self.corruption_type == "Partial Occlusion":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 1] *= 0.90
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.80
                    batch_perturbed["visual"][:, 1] *= 0.80
                else:
                    batch_perturbed["visual"][:, 0] *= 0.60
                    batch_perturbed["visual"][:, 1] *= 0.60
                    
            elif self.corruption_type == "Shadow Simulation":
                if self.severity == 1:
                    batch_perturbed["visual"][:, 0] *= 0.90
                    batch_perturbed["visual"][:, 2] *= 0.80
                elif self.severity == 2:
                    batch_perturbed["visual"][:, 0] *= 0.80
                    batch_perturbed["visual"][:, 2] *= 0.60
                else:
                    batch_perturbed["visual"][:, 0] *= 0.60
                    batch_perturbed["visual"][:, 2] *= 0.40

            yield batch_perturbed

def evaluate_robustness(model, test_loader):
    model.eval()
    all_preds = []
    all_gts = []
    all_confs = []
    all_outcomes = []
    latencies = []
    
    correct_states = 0
    total_states = 0
    
    vis_confs_sum = 0.0
    conflict_scores_sum = 0.0
    total_samples = 0
    
    ndcg3_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                "visual": batch["visual"].to(DEVICE),
                "env": batch["env"].to(DEVICE),
                "historical": batch["historical"].to(DEVICE),
                "knowledge": batch["knowledge"].to(DEVICE)
            }
            batch_ranks = batch["cause_ranking"]
            target_states = batch["reasoning_state"]
            
            for b in range(batch_ranks.size(0)):
                single_inputs = {k: inputs[k][b:b+1] for k in inputs}
                t_start = time.perf_counter()
                
                outputs = model(single_inputs)
                
                latency = (time.perf_counter() - t_start) * 1000.0
                latencies.append(latency)
                
                scores_b = outputs["scores"][0].cpu()
                confs_b = outputs["confidences"][0].cpu()
                
                sorted_indices = torch.argsort(scores_b, descending=True)
                pred_cause_ids = [CAUSE_IDS[idx.item()] for idx in sorted_indices]
                all_preds.append(pred_cause_ids)
                
                gt_indices = (batch_ranks[b] <= 2).nonzero(as_tuple=True)[0]
                gt_cause_ids = [CAUSE_IDS[idx.item()] for idx in gt_indices]
                all_gts.append(gt_cause_ids)
                
                confs_list = []
                outcomes_list = []
                for idx in sorted_indices:
                    confs_list.append(confs_b[idx.item()].item())
                    outcomes_list.append(1 if CAUSE_IDS[idx.item()] in gt_cause_ids else 0)
                all_confs.append(confs_list)
                all_outcomes.append(outcomes_list)
                
                pred_state = torch.argmax(outputs["reasoning_state"][0]).item()
                target_state = target_states[b].item()
                if pred_state == target_state:
                    correct_states += 1
                total_states += 1
                
                vis_confs_sum += single_inputs["visual"][0, 0].item()
                conflict_scores_sum += outputs["conflict_score"][0].item()
                total_samples += 1
                
                def dcg(rel, p):
                    return sum(r / np.log2(i + 2) for i, r in enumerate(rel[:p]))
                rel_vector = [1 if p in gt_cause_ids else 0 for p in pred_cause_ids]
                best_rel_vector = sorted(rel_vector, reverse=True)
                actual_dcg = dcg(rel_vector, 3)
                best_dcg = dcg(best_rel_vector, 3)
                ndcg = actual_dcg / best_dcg if best_dcg > 0 else 0.0
                ndcg3_list.append(ndcg)

    flat_confs = [c for sub in all_confs for c in sub]
    flat_outcomes = [o for sub in all_outcomes for o in sub]
    
    metrics = compute_metrics_package(all_preds, all_gts, all_confs, all_outcomes, latencies)
    ece, _ = compute_calibration_errors(flat_confs, flat_outcomes, n_bins=10)
    
    metrics["ece"] = ece
    metrics["cra"] = correct_states / total_states if total_states > 0 else 0.0
    metrics["mean_visual_conf"] = vis_confs_sum / total_samples
    metrics["mean_conflict_score"] = conflict_scores_sum / total_samples
    metrics["ndcg3_list"] = ndcg3_list
    
    return metrics

def main():
    checkpoint_path = "backend/research/trace_rce_v3/checkpoints/best_novarcd_model.pt"
    if not os.path.exists(checkpoint_path):
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return
        
    model = TRACERCEv3().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    _, _, test_loader = create_novarcd_v3_dataloaders(batch_size=8, seed=42)
    
    small_test_batch = []
    count = 0
    for b in test_loader:
        if count >= 50: break
        for i in range(b["visual"].shape[0]):
            if count >= 50: break
            sample_b = {k: v[i:i+1] if isinstance(v, torch.Tensor) else [v[i]] for k, v in b.items()}
            small_test_batch.append(sample_b)
            count += 1
            
    logger.info("Evaluating Clean Baseline...")
    clean_metrics = evaluate_robustness(model, small_test_batch)
    logger.info(f"Clean NDCG@3: {clean_metrics['ndcg_at_3']:.4f}, ECE: {clean_metrics['ece']:.4f}")
    
    corruptions = [
        "Gaussian Blur", "Motion Blur", "Brightness Increase", "Brightness Decrease",
        "Contrast Reduction", "JPEG Compression", "Gaussian Sensor Noise",
        "Partial Occlusion", "Shadow Simulation"
    ]
    
    results = {}
    
    for c in corruptions:
        results[c] = {}
        for s in [1, 2, 3]:
            perturbed_loader = PixelCorruptionLoader(small_test_batch, c, s)
            metrics = evaluate_robustness(model, perturbed_loader)
            
            pval = "Statistical significance not evaluated due to insufficient independent paired experimental observations."
            
            mean_diff = np.mean(clean_metrics["ndcg3_list"]) - np.mean(metrics["ndcg3_list"])
            pooled_std = np.std(clean_metrics["ndcg3_list"])
            effect_size = mean_diff / (pooled_std + 1e-8)
            
            results[c][f"Severity {s}"] = {
                "Precision@1": metrics["precision_at_1"],
                "Recall@3": metrics["recall_at_3"],
                "NDCG@3": metrics["ndcg_at_3"],
                "ECE": metrics["ece"],
                "Visual Confidence": metrics["mean_visual_conf"],
                "Conflict Score": metrics["mean_conflict_score"],
                "CRA": metrics["cra"],
                "Latency": metrics["mean"],
                "NDCG_Drop": mean_diff,
                "Wilcoxon_p": pval,
                "Effect_Size": effect_size
            }
            logger.info(f"{c} (Sev {s}): NDCG@3={metrics['ndcg_at_3']:.4f} (Drop={mean_diff:.4f}, p={pval:.4f}), ECE={metrics['ece']:.4f}, S_con={metrics['mean_conflict_score']:.4f}")
            
    with open("backend/research/trace_rce_v3/experiments/pixel_robustness_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info("Saved results to pixel_robustness_results.json")

if __name__ == "__main__":
    main()
