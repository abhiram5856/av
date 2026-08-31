"""
TRACE-RCE Statistical Validation Module
========================================
Phase 3.7 — Statistical Validation

Implements statistical analysis methods using numpy:
  - Bootstrap CI estimation (10,000 resamples)
  - Non-parametric permutation paired t-tests (exact p-values)
  - Cohen's d effect size calculations
  - Calibration curve binning & reliability table generators
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple


def bootstrap_metric_ci(
    scores: List[float],
    n_resamples: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes the mean and bootstrap confidence interval for a metric.
    Returns: (mean, lower_bound, upper_bound)
    """
    if not scores:
        return 0.0, 0.0, 0.0
        
    arr = np.array(scores)
    mean_val = float(np.mean(arr))
    
    # Bootstrap resampling
    resampled_means = []
    rng = np.random.default_rng(seed=42)  # set seed for reproducible validation
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        resampled_means.append(np.mean(sample))
        
    alpha = 1.0 - confidence_level
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)
    
    lower = float(np.percentile(resampled_means, lower_pct))
    upper = float(np.percentile(resampled_means, upper_pct))
    
    return mean_val, lower, upper


def compute_cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Computes Cohen's d effect size between two groups of equal size.
    d = (mean_a - mean_b) / pooled_std
    """
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    
    mean_a, mean_b = np.mean(arr_a), np.mean(arr_b)
    std_a, std_b = np.std(arr_a, ddof=1), np.std(arr_b, ddof=1)
    
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if pooled_std == 0.0:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


def paired_permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 2000
) -> Tuple[float, float]:
    """
    Performs a non-parametric paired permutation test.
    H0: The mean difference between scores_a and scores_b is 0.
    
    Returns:
        mean_difference: observed mean difference
        p_value: two-tailed permutation p-value
    """
    arr_a = np.array(scores_a)
    arr_b = np.array(scores_b)
    
    diffs = arr_a - arr_b
    observed_diff = np.mean(diffs)
    
    # Generate permuted differences by randomly flipping the sign of differences
    rng = np.random.default_rng(seed=42)
    perm_diffs = []
    
    for _ in range(n_permutations):
        # Randomly flip signs of differences (multiply by 1 or -1 with p=0.5)
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diffs.append(np.mean(diffs * signs))
        
    perm_diffs = np.array(perm_diffs)
    
    # Calculate two-tailed p-value
    extreme_count = np.sum(np.abs(perm_diffs) >= np.abs(observed_diff))
    p_value = float(extreme_count / n_permutations)
    
    return float(observed_diff), p_value


def generate_calibration_curve_data(
    confidences: List[float],
    outcomes: List[int],
    n_bins: int = 5
) -> List[Dict[str, Any]]:
    """
    Bins predicted confidences and computes accuracy within each bin.
    Used for rendering reliability diagrams.
    """
    bins = [[] for _ in range(n_bins)]
    
    # Sort confidences and outcomes into bins
    for c, o in zip(confidences, outcomes):
        bin_idx = min(int(c * n_bins), n_bins - 1)
        bins[bin_idx].append((c, o))
        
    curve_data = []
    for idx, b in enumerate(bins):
        bin_lower = idx / n_bins
        bin_upper = (idx + 1) / n_bins
        
        if not b:
            curve_data.append({
                "bin_range": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "avg_confidence": 0.0,
                "avg_accuracy": 0.0,
                "sample_size": 0
            })
            continue
            
        avg_conf = sum(p for p, _ in b) / len(b)
        avg_acc = sum(y for _, y in b) / len(b)
        
        curve_data.append({
            "bin_range": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
            "avg_confidence": round(float(avg_conf), 4),
            "avg_accuracy": round(float(avg_acc), 4),
            "sample_size": len(b)
        })
        
    return curve_data
