"""
TRACE-RCE Error Analysis Module
===============================
Phase 3.6 — Error Analysis

For every evaluation case where TRACE-RCE's top-1 predicted cause is
incorrect (i.e. not in the ground-truth cause set):
  - Resolves why it failed
  - Identifies the dominant modality that skewed the prediction
  - Quantifies missing evidence for the actual ground-truth cause
  - Categorizes the failure mode
"""

from __future__ import annotations
from typing import List, Dict, Any
from backend.research.root_cause_engine.dataset import EvalScenario
from backend.research.root_cause_engine.models import RootCauseResult, RootCause


def analyze_prediction_errors(
    scenarios: List[EvalScenario],
    results: List[RootCauseResult]
) -> List[Dict[str, Any]]:
    """
    Scans the results, finds failures where top prediction is incorrect,
    and performs deep-dive error diagnostic checks.
    """
    results_map = {r.context_hash: r for r in results}
    failures = []

    for scen in scenarios:
        h = scen.context.get_content_hash()
        res = results_map.get(h)
        if not res or not res.ranked_causes:
            continue

        top_predicted_cause = res.ranked_causes[0].cause_id
        gt_causes = scen.ground_truth_causes

        # A failure occurs if the top-1 predicted cause is not in ground truth
        if top_predicted_cause not in gt_causes:
            # Diagnose failure
            top_rc: RootCause = res.ranked_causes[0]
            
            # 1. Determine dominant modality for the incorrect prediction
            dominant_mod = "none"
            max_contrib = -1.0
            for contrib in top_rc.evidence_contributions:
                if contrib.weighted_contribution > max_contrib:
                    max_contrib = contrib.weighted_contribution
                    dominant_mod = contrib.modality

            # 2. Analyze why ground-truth cause was missed (missing evidence)
            # Find where the ground-truth cause was ranked
            gt_rc_match = None
            gt_rank = -1
            for rc in res.ranked_causes:
                if rc.cause_id in gt_causes:
                    gt_rc_match = rc
                    gt_rank = rc.rank
                    break

            missing_modality_contributions = {}
            if gt_rc_match:
                for contrib in gt_rc_match.evidence_contributions:
                    # Record scores of ground-truth cause to see where it fell short
                    missing_modality_contributions[contrib.modality] = {
                        "raw_score": contrib.raw_evidence_score,
                        "weighted": contrib.weighted_contribution
                    }

            # 3. Categorize failure mode
            meta = scen.metadata
            pert = meta.get("perturbation", "none")
            
            if pert == "contradictory_evidence":
                failure_category = "Contradictory Evidence Confounding"
            elif pert == "missing_weather":
                failure_category = "Missing Environmental Modality"
            elif pert == "missing_rag":
                failure_category = "Missing Knowledge Modality"
            elif pert == "low_confidence":
                failure_category = "Low Vision Model Confidence"
            elif pert == "ood_disease":
                failure_category = "Out-Of-Distribution Target Disease"
            elif scen.context.image.is_valid_quality is False:
                failure_category = "Degraded Visual Signal Quality"
            else:
                failure_category = "Evidence Ambiguity / Multi-factorial Overlap"

            failures.append({
                "case_id": scen.case_id,
                "description": scen.description,
                "predicted_top": top_predicted_cause,
                "predicted_top_cas": top_rc.cas_final,
                "ground_truth": gt_causes,
                "dominant_modality": dominant_mod,
                "dominant_contribution": round(max_contrib, 4),
                "actual_cause_rank": gt_rank if gt_rank != -1 else "unranked (> threshold)",
                "actual_cause_contributions": missing_modality_contributions,
                "failure_mode": failure_category,
                "explanation": (
                    f"Predicted '{top_predicted_cause}' driven by '{dominant_mod}' (weighted contribution={max_contrib:.2f}). "
                    f"The actual ground truth causes were {gt_causes}. "
                    f"Failure class: {failure_category}."
                )
            })

    return failures


def generate_error_summary_report(failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates counts and percentages of errors by failure mode and dominant modality."""
    if not failures:
        return {"total_failures": 0}

    failure_modes = {}
    dominant_modalities = {}

    for f in failures:
        fm = f["failure_mode"]
        dm = f["dominant_modality"]

        failure_modes[fm] = failure_modes.get(fm, 0) + 1
        dominant_modalities[dm] = dominant_modalities.get(dm, 0) + 1

    total = len(failures)
    return {
        "total_failures": total,
        "failure_modes_distribution": {
            k: {"count": v, "percentage": round((v / total) * 100, 2)}
            for k, v in failure_modes.items()
        },
        "dominant_modalities_distribution": {
            k: {"count": v, "percentage": round((v / total) * 100, 2)}
            for k, v in dominant_modalities.items()
        }
    }
