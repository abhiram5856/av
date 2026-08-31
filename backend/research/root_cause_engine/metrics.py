"""
TRACE-RCE Evaluation Metrics Engine
===================================
Phase 3.3 — Metrics

Implements statistical metrics for evaluating TRACE-RCE and baselines:
  - Precision@1, Precision@3
  - Recall (specifically Recall@3)
  - Mean Reciprocal Rank (MRR)
  - NDCG@3 (Normalized Discounted Cumulative Gain at 3)
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Brier Score
  - Latency percentiles (mean, p50, p95, p99)
  - Determinism rate
  - Evidence completeness
  - Conflict detection rate
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Tuple


def compute_precision_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """Precision@K: fraction of top-K predictions in ground truth."""
    if not predicted:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in ground_truth)
    return hits / k


def compute_recall_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """Recall@K: fraction of ground truth causes found in top-K predictions."""
    if not ground_truth:
        return 1.0
    top_k = predicted[:k]
    hits = sum(1 for p in top_k if p in ground_truth)
    return hits / len(ground_truth)


def compute_mrr(predicted: List[str], ground_truth: List[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first correct match."""
    for rank, p in enumerate(predicted, start=1):
        if p in ground_truth:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(predicted: List[str], ground_truth: List[str], k: int) -> float:
    """NDCG@K with binary relevance (1 if in ground truth, 0 otherwise)."""
    if not predicted or not ground_truth:
        return 0.0
        
    top_k = predicted[:k]
    dcg = 0.0
    for idx, p in enumerate(top_k):
        rel = 1.0 if p in ground_truth else 0.0
        dcg += rel / math.log2(idx + 2)
        
    # Ideal DCG: top min(k, len(ground_truth)) elements are all hits
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, len(ground_truth))))
    
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Brier Score: Mean squared error of calibrated confidence predictions."""
    if not predictions:
        return 0.0
    errors = [(p - y) ** 2 for p, y in zip(predictions, outcomes)]
    return float(np.mean(errors))


def compute_calibration_errors(
    predictions: List[float], outcomes: List[int], n_bins: int = 5
) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    """
    if not predictions:
        return 0.0, 0.0

    bins = [[] for _ in range(n_bins)]
    for p, y in zip(predictions, outcomes):
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bins[bin_idx].append((p, y))

    ece = 0.0
    mce = 0.0
    total = len(predictions)

    for b in bins:
        if b:
            avg_conf = sum(p for p, _ in b) / len(b)
            avg_acc = sum(y for _, y in b) / len(b)
            err = abs(avg_conf - avg_acc)
            ece += (len(b) / total) * err
            mce = max(mce, err)

    return float(ece), float(mce)


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Computes mean, p50, p95, and p99 from a list of latencies in ms."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def compute_metrics_package(
    all_predictions: List[List[str]],
    all_ground_truths: List[List[str]],
    all_confidences: List[List[float]] = None,
    all_outcomes: List[List[int]] = None,
    latencies: List[float] = None,
    conflicts_detected: List[bool] = None,
    completeness_scores: List[float] = None,
    determinism_results: List[bool] = None,
) -> Dict[str, Any]:
    """
    Computes the complete experimental validation package.
    """
    n = len(all_predictions)
    if n == 0:
        return {}

    p1 = [compute_precision_at_k(p, gt, 1) for p, gt in zip(all_predictions, all_ground_truths)]
    p3 = [compute_precision_at_k(p, gt, 3) for p, gt in zip(all_predictions, all_ground_truths)]
    r3 = [compute_recall_at_k(p, gt, 3) for p, gt in zip(all_predictions, all_ground_truths)]
    mrr = [compute_mrr(p, gt) for p, gt in zip(all_predictions, all_ground_truths)]
    ndcg3 = [compute_ndcg_at_k(p, gt, 3) for p, gt in zip(all_predictions, all_ground_truths)]

    result = {
        "precision_at_1": float(np.mean(p1)),
        "precision_at_3": float(np.mean(p3)),
        "recall_at_3": float(np.mean(r3)),
        "mrr": float(np.mean(mrr)),
        "ndcg_at_3": float(np.mean(ndcg3)),
    }

    # Add calibration metrics if provided
    if all_confidences is not None and all_outcomes is not None:
        flat_conf = [c for sub in all_confidences for c in sub]
        flat_out = [o for sub in all_outcomes for o in sub]
        ece, mce = compute_calibration_errors(flat_conf, flat_out)
        brier = compute_brier_score(flat_conf, flat_out)
        result.update({
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "brier_score": brier
        })

    # Add latency metrics if provided
    if latencies:
        result.update(compute_latency_percentiles(latencies))

    # Add auxiliary evaluation properties
    if conflicts_detected:
        result["conflict_detection_rate"] = float(np.mean(conflicts_detected))
    if completeness_scores:
        result["evidence_completeness"] = float(np.mean(completeness_scores))
    if determinism_results:
        result["determinism_rate"] = float(np.mean(determinism_results))

    return result
