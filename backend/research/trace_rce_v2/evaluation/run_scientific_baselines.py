import os
import sys
import json
import time
import numpy as np
import torch
import asyncio
from typing import List, Dict, Any
from collections import Counter

# Add root directory to sys path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.dataset.dataloader import rce_collate_fn
from backend.research.trace_rce_v2.dataset.novarcd_dataset import load_split_records, deserialize_context, create_novarcd_dataloaders
from backend.research.trace_rce_v2.dataset.feature_extractor import extract_features
from backend.research.root_cause_engine.metrics import compute_metrics_package
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.evaluation.evaluator import run_v1_evaluation, run_v2_evaluation, paired_bootstrap_test
from backend.research.root_cause_engine.dataset import EvalScenario

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints\best_novarcd_model.pt"

def extract_flat_features(records):
    X = []
    y = []
    
    for rec in records:
        ctx = deserialize_context(rec["context"])
        features = extract_features(ctx)
        
        # Flatten all modality features into a single 1D array
        flat_f = np.concatenate([
            features.visual,
            features.env,
            features.severity,
            features.knowledge,
            features.historical
        ])
        X.append(flat_f)
        
        gt_causes = rec["ground_truth_causes"]
        if not gt_causes:
            primary_cause_idx = 0
        else:
            primary_cause_idx = CAUSE_IDS.index(gt_causes[0])
            
        y.append(primary_cause_idx)
        
    return np.array(X), np.array(y)

def run_sklearn_baseline(model, X_train, y_train, X_test, test_records, name="Baseline"):
    print(f"Training {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    
    t1 = time.time()
    probs = model.predict_proba(X_test)
    inference_time = (time.time() - t1) / len(X_test) * 1000.0 # ms per sample
    
    all_preds = []
    all_gts = []
    all_confs = []
    all_outcomes = []
    
    for i, rec in enumerate(test_records):
        gt_causes = rec["ground_truth_causes"]
        all_gts.append(gt_causes)
        
        sample_probs = probs[i]
        
        # Sort indices by probability
        sorted_indices = np.argsort(sample_probs)[::-1]
        pred_causes = [CAUSE_IDS[idx] for idx in sorted_indices]
        all_preds.append(pred_causes)
        
        confs = []
        outcomes = []
        for idx in sorted_indices:
            confs.append(sample_probs[idx])
            outcomes.append(1 if CAUSE_IDS[idx] in gt_causes else 0)
            
        all_confs.append(confs)
        all_outcomes.append(outcomes)
        
    metrics = compute_metrics_package(
        all_predictions=all_preds,
        all_ground_truths=all_gts,
        all_confidences=all_confs,
        all_outcomes=all_outcomes,
        latencies=[inference_time] * len(X_test)
    )
    
    return metrics, {"predictions": all_preds, "ground_truths": all_gts, "confidences": all_confs, "outcomes": all_outcomes}

def run_heuristic_baselines(train_records, test_records):
    # Majority Baseline
    all_train_causes = [r["ground_truth_causes"][0] if r["ground_truth_causes"] else CAUSE_IDS[0] for r in train_records]
    cause_counts = Counter(all_train_causes)
    majority_ranked = [c for c, _ in cause_counts.most_common()]
    
    # Fill remaining causes that might not be in train
    for cid in CAUSE_IDS:
        if cid not in majority_ranked:
            majority_ranked.append(cid)
            
    majority_preds = []
    majority_gts = []
    majority_confs = []
    majority_outcomes = []
    
    random_preds = []
    random_confs = []
    random_outcomes = []
    
    for rec in test_records:
        gt = rec["ground_truth_causes"]
        majority_gts.append(gt)
        
        # Majority
        majority_preds.append(majority_ranked.copy())
        m_conf = []
        m_out = []
        for idx, cid in enumerate(majority_ranked):
            m_conf.append(1.0 / (idx + 1)) # dummy confidence
            m_out.append(1 if cid in gt else 0)
        majority_confs.append(m_conf)
        majority_outcomes.append(m_out)
        
        # Random
        rand_ranked = list(CAUSE_IDS)
        np.random.shuffle(rand_ranked)
        random_preds.append(rand_ranked)
        r_conf = []
        r_out = []
        for idx, cid in enumerate(rand_ranked):
            r_conf.append(1.0 / (idx + 1))
            r_out.append(1 if cid in gt else 0)
        random_confs.append(r_conf)
        random_outcomes.append(r_out)
        
    majority_metrics = compute_metrics_package(majority_preds, majority_gts, majority_confs, majority_outcomes, [1.0]*len(test_records))
    random_metrics = compute_metrics_package(random_preds, majority_gts, random_confs, random_outcomes, [1.0]*len(test_records))
    
    return majority_metrics, random_metrics

async def main():
    print("========================================")
    print("NOVA-RCD: Comprehensive Baseline Evaluation")
    print("========================================")
    
    train_records, val_records, test_records = load_split_records()
    
    print("Extracting flattened features for Linear/MLP Baselines...")
    X_train, y_train = extract_flat_features(train_records)
    X_test, y_test = extract_flat_features(test_records)
    
    # 1. Heuristics
    print("Running Heuristic Baselines...")
    maj_metrics, rand_metrics = run_heuristic_baselines(train_records, test_records)
    
    # 2. Linear (Logistic Regression)
    lr = LogisticRegression(max_iter=1000)
    lr_metrics, lr_raw = run_sklearn_baseline(lr, X_train, y_train, X_test, test_records, "Logistic Regression")
    
    # 3. MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mlp_metrics, mlp_raw = run_sklearn_baseline(mlp, X_train, y_train, X_test, test_records, "MLP")
    
    # 4. TRACE-RCE v1
    print("Evaluating TRACE-RCE v1 (Rule-Based)...")
    test_scenarios = [EvalScenario(context=deserialize_context(r["context"]), ground_truth_causes=r["ground_truth_causes"], case_id=r["case_id"], description="") for r in test_records]
    v1_metrics, v1_raw = await run_v1_evaluation(test_scenarios)
    
    # 5. TRACE-RCE v2
    print("Evaluating TRACE-RCE v2 (Neural Causal Engine)...")
    _, _, test_loader = create_novarcd_dataloaders(batch_size=32, augmentation_factor=0)
    model = build_model()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.to(DEVICE)
    
    v2_metrics, v2_raw = run_v2_evaluation(model, test_loader, DEVICE)
    
    print("\n[Metrics Comparison]")
    print(f"{'Model':<20} | {'NDCG@3':<8} | {'MRR':<8} | {'P@1':<8} | {'R@3':<8} | {'ECE':<8}")
    print("-" * 70)
    print(f"{'Random Ranking':<20} | {rand_metrics['ndcg_at_3']:.4f}   | {rand_metrics['mrr']:.4f}   | {rand_metrics['precision_at_1']:.4f}   | {rand_metrics['recall_at_3']:.4f}   | N/A")
    print(f"{'Majority Class':<20} | {maj_metrics['ndcg_at_3']:.4f}   | {maj_metrics['mrr']:.4f}   | {maj_metrics['precision_at_1']:.4f}   | {maj_metrics['recall_at_3']:.4f}   | N/A")
    print(f"{'Logistic Regression':<20} | {lr_metrics['ndcg_at_3']:.4f}   | {lr_metrics['mrr']:.4f}   | {lr_metrics['precision_at_1']:.4f}   | {lr_metrics['recall_at_3']:.4f}   | {lr_metrics['expected_calibration_error']:.4f}")
    print(f"{'MLP (No Attention)':<20} | {mlp_metrics['ndcg_at_3']:.4f}   | {mlp_metrics['mrr']:.4f}   | {mlp_metrics['precision_at_1']:.4f}   | {mlp_metrics['recall_at_3']:.4f}   | {mlp_metrics['expected_calibration_error']:.4f}")
    print(f"{'TRACE-RCE v1':<20} | {v1_metrics['ndcg_at_3']:.4f}   | {v1_metrics['mrr']:.4f}   | {v1_metrics['precision_at_1']:.4f}   | {v1_metrics['recall_at_3']:.4f}   | {v1_metrics['expected_calibration_error']:.4f}")
    print(f"{'TRACE-RCE v2':<20} | {v2_metrics['ndcg_at_3']:.4f}   | {v2_metrics['mrr']:.4f}   | {v2_metrics['precision_at_1']:.4f}   | {v2_metrics['recall_at_3']:.4f}   | {v2_metrics['expected_calibration_error']:.4f}")
    
    print("\nRunning Statistical Significance Testing (V1 vs V2)...")
    sig_results = paired_bootstrap_test(v1_raw, v2_raw)
    print(f"NDCG@3 p-value: {sig_results['ndcg_p_value']:.6f}")
    print(f"Brier p-value:  {sig_results['brier_p_value']:.6f}")
    
    # Save the full results for report generation later
    with open("scientific_baseline_results.json", "w") as f:
        json.dump({
            "Random": rand_metrics,
            "Majority": maj_metrics,
            "Logistic Regression": lr_metrics,
            "MLP (No Attention)": mlp_metrics,
            "TRACE-RCE v1": v1_metrics,
            "TRACE-RCE v2": v2_metrics,
            "Significance": sig_results
        }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
