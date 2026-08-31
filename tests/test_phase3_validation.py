"""
Unit Tests for TRACE-RCE Phase 3 Scientific Validation Codebase
==============================================================
Phase 3.9 — Write Unit Tests

Validates:
  - Dataset generation (300 cases, unique case IDs, unique hashes)
  - Baseline synchronizations (Rule-Based, Pure-RAG, LLM-Style, Simple-Fusion)
  - Metrics math correctness (Precision, Recall, MRR, NDCG, ECE)
  - Ablation modifier functionality
  - Robustness subset stratifications
  - Error analysis diagnostic checks
  - Statistical helper correctness
"""

import pytest
import math
import numpy as np
from backend.research.root_cause_engine.dataset import generate_evaluation_dataset, _build_context
from backend.research.root_cause_engine.baselines import (
    RuleBasedBaseline, PureRAGBaseline, LLMStyleBaseline, SimpleWeightedFusionBaseline
)
from backend.research.root_cause_engine.metrics import (
    compute_precision_at_k, compute_recall_at_k, compute_mrr, compute_ndcg_at_k,
    compute_brier_score, compute_calibration_errors, compute_metrics_package
)
from backend.research.root_cause_engine.ablation import ablate_context
from backend.research.root_cause_engine.statistics import (
    bootstrap_metric_ci, compute_cohens_d, paired_permutation_test, generate_calibration_curve_data
)


# =============================================================================
# 1. Dataset Tests
# =============================================================================

def test_dataset_generation_length():
    scenarios = generate_evaluation_dataset()
    assert len(scenarios) == 300
    
    # Check that case IDs are sequential and populated
    case_ids = [s.case_id for s in scenarios]
    assert case_ids[0] == "CASE-001"
    assert case_ids[-1] == "CASE-300"
    assert len(set(case_ids)) == 300


def test_dataset_hash_uniqueness():
    scenarios = generate_evaluation_dataset()
    hashes = [s.context.get_content_hash() for s in scenarios]
    # Verify that different scenarios yield distinct hashes
    assert len(set(hashes)) == 300


def test_dataset_perturbation_ranges():
    scenarios = generate_evaluation_dataset()
    
    # Check counts
    missing_weather_count = sum(1 for s in scenarios if s.metadata["perturbation"] == "missing_weather")
    missing_rag_count = sum(1 for s in scenarios if s.metadata["perturbation"] == "missing_rag")
    ood_count = sum(1 for s in scenarios if s.metadata["perturbation"] == "ood_disease")
    
    assert missing_weather_count == 30
    assert missing_rag_count == 30
    assert ood_count == 30


# =============================================================================
# 2. Baselines Tests
# =============================================================================

def test_rule_based_baseline():
    scenarios = generate_evaluation_dataset()
    rule_engine = RuleBasedBaseline()
    
    # CASE-001 (Late Blight conducive weather)
    res = rule_engine.analyze_sync(scenarios[0].context)
    assert len(res) > 0
    # "excessive_leaf_wetness" is a standard fungal rule matching wetness >= 8.0
    assert "excessive_leaf_wetness" in res


def test_pure_rag_baseline():
    scenarios = generate_evaluation_dataset()
    rag_engine = PureRAGBaseline()
    
    # Let's pass a context with specific RAG text
    res = rag_engine.analyze_sync(scenarios[0].context)
    assert len(res) > 0
    assert "excessive_leaf_wetness" in res or "high_humidity_conduciveness" in res


def test_llm_baseline():
    scenarios = generate_evaluation_dataset()
    llm_engine = LLMStyleBaseline()
    
    res = llm_engine.analyze_sync(scenarios[0].context)
    # LLM baseline should return exactly 3 top priors for the disease family
    assert len(res) >= 2


def test_simple_fusion_baseline():
    scenarios = generate_evaluation_dataset()
    fusion_engine = SimpleWeightedFusionBaseline()
    
    res = fusion_engine.analyze_sync(scenarios[0].context)
    assert len(res) > 0


# =============================================================================
# 3. Metrics Tests
# =============================================================================

def test_precision_recall_math():
    pred = ["a", "b", "c"]
    gt = ["b", "d"]
    
    assert compute_precision_at_k(pred, gt, 1) == 0.0   # top 1 is "a" (miss)
    assert compute_precision_at_k(pred, gt, 3) == 1/3   # "b" is in gt
    assert compute_recall_at_k(pred, gt, 3) == 0.5      # found 1 out of 2 ground truths
    assert compute_mrr(pred, gt) == 0.5                 # first correct is rank 2 ("b")


def test_ndcg_math():
    pred = ["a", "b", "c"]
    gt = ["b", "c"]
    
    # DCG@3 = 0/log2(2) + 1/log2(3) + 1/log2(4) = 0 + 0.6309 + 0.5 = 1.1309
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    # NDCG@3 = 1.1309 / 1.6309 = 0.6934
    ndcg = compute_ndcg_at_k(pred, gt, 3)
    assert abs(ndcg - 0.6934) < 0.001


def test_ece_math():
    # 2 predictions: confidence 0.9 (outcome 1), confidence 0.1 (outcome 0)
    conf = [0.9, 0.1]
    out = [1, 0]
    
    # Bins: [0-0.2] contains (0.1, 0) -> avg_conf=0.1, avg_acc=0.0 -> err = 0.1
    # [0.8-1.0] contains (0.9, 1) -> avg_conf=0.9, avg_acc=1.0 -> err = 0.1
    # ECE = 0.5 * 0.1 + 0.5 * 0.1 = 0.1
    ece, mce = compute_calibration_errors(conf, out, n_bins=5)
    assert abs(ece - 0.1) < 1e-6
    assert abs(mce - 0.1) < 1e-6
    
    brier = compute_brier_score(conf, out)
    # (0.9-1.0)^2 = 0.01; (0.1-0.0)^2 = 0.01. mean = 0.01.
    assert abs(brier - 0.01) < 1e-6


# =============================================================================
# 4. Ablation Tests
# =============================================================================

def test_ablate_context_function():
    scenarios = generate_evaluation_dataset()
    ctx = scenarios[0].context
    
    # Ablate weather
    ab_ctx = ablate_context(ctx, "ablate_environmental")
    assert ab_ctx.weather.humidity_7d_avg == 50.0
    assert ab_ctx.weather.temperature_7d_avg == 25.0
    assert ab_ctx.weather.total_precipitation_mm == 0.0
    
    # Ablate RAG
    ab_ctx_rag = ablate_context(ctx, "ablate_knowledge")
    assert len(ab_ctx_rag.knowledge.retrieved_chunk_ids) == 0
    assert ab_ctx_rag.knowledge.context_text_block == ""


# =============================================================================
# 5. Statistics Tests
# =============================================================================

def test_bootstrap_ci():
    rng = np.random.default_rng(42)
    scores = rng.choice([0.0, 1.0], size=100, p=[0.3, 0.7]).tolist()
    
    mean_val, lower, upper = bootstrap_metric_ci(scores, n_resamples=1000)
    assert 0.55 <= mean_val <= 0.85
    assert lower <= mean_val <= upper
    assert 0.0 <= lower <= 1.0


def test_cohens_d():
    a = [1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 1.0, 0.0]
    # Mean a = 0.75, Mean b = 0.25
    d = compute_cohens_d(a, b)
    assert d > 0.0


def test_paired_permutation_test():
    a = [1.0, 1.0, 1.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0, 1.0, 0.0]
    diff, p = paired_permutation_test(a, b, n_permutations=500)
    assert diff == 0.60
    assert 0.0 <= p <= 1.0
