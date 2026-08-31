"""
TRACE-RCE Robustness Evaluator
==============================
Phase 3.5 — Robustness

Groups and analyzes TRACE-RCE performance across different perturbation
subsets from the generated evaluation dataset:
  - Standard (no perturbations)
  - Missing Weather
  - Missing RAG
  - Contradictory Evidence
  - Low Confidence
  - Out of Distribution (OOD) Diseases
  - Blurry/Invalid Images (isolated from image quality metadata)
  - No History (isolated from history depth metadata)
"""

from __future__ import annotations
from typing import List, Dict, Any
from backend.research.root_cause_engine.dataset import EvalScenario
from backend.research.root_cause_engine.models import RootCauseResult
from backend.research.root_cause_engine.metrics import compute_metrics_package


def run_robustness_analysis(
    scenarios: List[EvalScenario],
    results: List[RootCauseResult]
) -> Dict[str, Dict[str, Any]]:
    """
    Computes Precision@1, Precision@3, Recall@3, and Latency stratified
    by perturbation categories.
    """
    # Create lookup for results by case_id
    results_map = {r.context_hash: r for r in results}

    # Helper to resolve scenario result
    def get_res(scen: EvalScenario) -> RootCauseResult:
        h = scen.context.get_content_hash()
        return results_map[h]

    # Categorize cases
    categories = {
        "standard": [],
        "missing_weather": [],
        "missing_rag": [],
        "contradictory_evidence": [],
        "low_confidence": [],
        "ood_disease": [],
        "blurry_images": [],
        "zero_history": []
    }

    for scen in scenarios:
        meta = scen.metadata
        pert = meta.get("perturbation", "none")
        res = get_res(scen)
        if not res:
            continue

        # Map to specific categories
        if pert == "none":
            categories["standard"].append((scen, res))
        elif pert == "missing_weather":
            categories["missing_weather"].append((scen, res))
        elif pert == "missing_rag":
            categories["missing_rag"].append((scen, res))
        elif pert == "contradictory_evidence":
            categories["contradictory_evidence"].append((scen, res))
        elif pert == "low_confidence":
            categories["low_confidence"].append((scen, res))
        elif pert == "ood_disease":
            categories["ood_disease"].append((scen, res))

        # Check isolated cases across all data
        if scen.context.image.is_valid_quality is False:
            categories["blurry_images"].append((scen, res))
        if scen.context.history.total_previous_diagnoses == 0:
            categories["zero_history"].append((scen, res))

    # Compute metrics per category
    analysis = {}
    for cat_name, pairs in categories.items():
        if not pairs:
            analysis[cat_name] = {
                "count": 0,
                "precision_at_1": 0.0,
                "precision_at_3": 0.0,
                "recall_at_3": 0.0,
                "avg_latency_ms": 0.0
            }
            continue

        preds = [[rc.cause_id for rc in r.ranked_causes] for _, r in pairs]
        gts = [s.ground_truth_causes for s, _ in pairs]
        lats = [r.analysis_latency_ms for _, r in pairs if r.analysis_latency_ms is not None]

        pkg = compute_metrics_package(preds, gts, latencies=lats)
        analysis[cat_name] = {
            "count": len(pairs),
            "precision_at_1": pkg.get("precision_at_1", 0.0),
            "precision_at_3": pkg.get("precision_at_3", 0.0),
            "recall_at_3": pkg.get("recall_at_3", 0.0),
            "avg_latency_ms": pkg.get("mean", 0.0)
        }

    return analysis
