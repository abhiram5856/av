"""
TRACE-RCE — Main Engine
========================
Temporal-Relational Adaptive Causal Explainer — Root Cause Engine

This module implements the AbstractRootCauseEngine interface defined in
backend/services/root_cause_service.py.

TRACE-RCE is the primary original research contribution for NOVA v2.0.
It extends the NOVA v1.0 production system without modifying any production module.

Algorithm Pipeline:
    1. Evidence Extraction (EEM)           — per-modality evidence vectors
    2. Reliability Scoring                 — dynamic weight adjustment
    3. Adaptive Evidence Fusion (AEF)      — weighted CAS_raw per hypothesis
    4. Conflict Resolution (CRL)           — detect and discount contradictions
    5. Confidence Calibration (CCM)        — Platt calibration + uncertainty
    6. Causal Ranking (CRE)               — sorted ranked causes
    7. Intervention Priority Scheduling    — IPS-sorted action plan
    8. Reasoning Chain + Evidence Graph   — explainability artifacts

Reference: TRACE-RCE specification in nova_rce_research.md
"""

from __future__ import annotations
import time
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from backend.schemas.context import AIContext
from backend.services.root_cause_service import AbstractRootCauseEngine

from backend.research.root_cause_engine.models import (
    RootCauseResult, ConflictEvent
)
from backend.research.root_cause_engine.ontology import (
    get_hypotheses_for_disease, classify_disease_family
)
from backend.research.root_cause_engine.evidence import (
    extract_visual_evidence,
    extract_environmental_evidence,
    extract_concern_evidence,
    extract_knowledge_evidence,
    extract_historical_evidence,
    compute_reliability_scores,
    PRIOR_MODALITY_WEIGHTS,
)
from backend.research.root_cause_engine.fusion import (
    compute_reliability_adjusted_weights,
    compute_cas_raw,
    detect_conflicts,
    apply_conflict_discounting,
    calibrate_confidence,
    _MIN_THRESHOLD,
)
from backend.research.root_cause_engine.ranking import (
    build_root_cause,
    build_evidence_graph,
    build_reasoning_chain,
    generate_risk_forecast,
)

logger = logging.getLogger("nova.research.trace_rce")


class TRACERootCauseEngine(AbstractRootCauseEngine):
    """
    TRACE-RCE: Temporal-Relational Adaptive Causal Explainer — Root Cause Engine

    Implements AbstractRootCauseEngine from the production service interface.
    Consumes a sealed, immutable AIContext object and produces a RootCauseResult.

    This class is the entry point for all external integrations.
    """

    def __init__(self, min_threshold: float = _MIN_THRESHOLD):
        """
        Args:
            min_threshold: Minimum CAS_final score for a cause to be included in output.
                           Default = 0.15. Can be tuned for sensitivity/specificity tradeoff.
        """
        self.min_threshold = min_threshold
        self._analysis_store: Dict[str, RootCauseResult] = {}
        logger.info("TRACE-RCE initialized (min_threshold=%.2f)", min_threshold)

    async def analyze(self, context: AIContext) -> RootCauseResult:
        """
        Execute the full TRACE-RCE pipeline on the given AIContext.

        Returns:
            RootCauseResult with ranked causes, evidence graph, reasoning chain,
            confidence intervals, and intervention priorities.
        """
        t_start = time.perf_counter()
        analysis_id = uuid4()

        disease_name = context.vision.predicted_disease
        disease_family = classify_disease_family(disease_name)
        context_hash = context.get_content_hash()

        logger.info(
            "[TRACE-RCE] Analyzing: disease=%s family=%s ctx_hash=%s",
            disease_name, disease_family, context_hash[:12]
        )

        # ── Step 1: Load Causal Hypothesis Set ──────────────────────────────
        hypotheses = get_hypotheses_for_disease(disease_name)
        logger.debug("[TRACE-RCE] Loaded %d causal hypotheses for %s", len(hypotheses), disease_family)

        # ── Step 2: Compute Modality Reliability Scores ──────────────────────
        reliability = compute_reliability_scores(context)
        adjusted_weights = compute_reliability_adjusted_weights(PRIOR_MODALITY_WEIGHTS, reliability)
        logger.debug("[TRACE-RCE] Reliability: %s", reliability)

        # ── Step 3: Extract Evidence + Score All Hypotheses ──────────────────
        all_conflicts: List[ConflictEvent] = []
        scored_hypotheses = []
        evidence_scores_per_cause: Dict[str, Dict[str, float]] = {}

        for cause_id, hyp in hypotheses.items():
            e_scores = {
                "visual":        extract_visual_evidence(context, hyp),
                "environmental": extract_environmental_evidence(context, hyp),
                "severity":      extract_concern_evidence(context, hyp),
                "knowledge":     extract_knowledge_evidence(context, hyp),
                "historical":    extract_historical_evidence(context, hyp),
            }
            evidence_scores_per_cause[cause_id] = e_scores

            # ── AEF: Compute CAS_raw ─────────────────────────────────────────
            cas_raw = compute_cas_raw(e_scores, adjusted_weights)

            # ── CRL: Conflict Detection + Discounting ────────────────────────
            conflicts, mean_conflict = detect_conflicts(e_scores, cause_id)
            cas_final, discount_applied = apply_conflict_discounting(cas_raw, mean_conflict)

            for c in conflicts:
                all_conflicts.append(ConflictEvent(
                    modality_a=c["modality_a"],
                    modality_b=c["modality_b"],
                    cause_id=c["cause_id"],
                    conflict_score=c["conflict_score"],
                    discounting_applied=round(discount_applied, 4),
                ))

            # ── CCM: Confidence Calibration ──────────────────────────────────
            confidence_dict = calibrate_confidence(
                cas_final=cas_final,
                evidence_scores=e_scores,
                adjusted_weights=adjusted_weights,
                reliability=reliability,
                vision_confidence=context.vision.confidence_score,
                is_valid_image=context.image.is_valid_quality,
            )

            scored_hypotheses.append({
                "hypothesis": hyp,
                "e_scores": e_scores,
                "cas_raw": cas_raw,
                "cas_final": cas_final,
                "discount": discount_applied,
                "confidence": confidence_dict,
            })

        # ── Step 4: Filter and Rank ──────────────────────────────────────────
        above_threshold = [s for s in scored_hypotheses if s["cas_final"] >= self.min_threshold]
        above_threshold.sort(key=lambda x: x["cas_final"], reverse=True)

        # ── Step 5: Build RootCause Objects ──────────────────────────────────
        ranked_causes = []
        for rank_idx, s in enumerate(above_threshold, start=1):
            rc = build_root_cause(
                rank=rank_idx,
                hypothesis=s["hypothesis"],
                disease_family=disease_family,
                evidence_scores=s["e_scores"],
                adjusted_weights=adjusted_weights,
                cas_raw=s["cas_raw"],
                cas_final=s["cas_final"],
                conflict_discount=s["discount"],
                confidence_dict=s["confidence"],
            )
            ranked_causes.append(rc)

        # ── Step 6: Build Explainability Artifacts ────────────────────────────
        reasoning_chain = build_reasoning_chain(disease_name, disease_family, ranked_causes)

        evidence_graph = build_evidence_graph(
            disease_name=disease_name,
            disease_family=disease_family,
            ranked_causes=ranked_causes,
            evidence_scores_per_cause=evidence_scores_per_cause,
            adjusted_weights=adjusted_weights,
        )

        # ── Step 7: Top Intervention + Risk Forecast ──────────────────────────
        top_cause = ranked_causes[0] if ranked_causes else None
        top_intervention = (
            top_cause.actionable_steps[0] if top_cause and top_cause.actionable_steps else None
        )

        risk_forecast = generate_risk_forecast(
            disease_family=disease_family,
            final_severity=context.concern.concern_score,
            top_cause=top_cause,
            humidity=context.weather.humidity_7d_avg,
            temperature=context.weather.temperature_7d_avg,
        )

        # ── Step 8: Assemble Final Result ─────────────────────────────────────
        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000, 2)

        result = RootCauseResult(
            analysis_id=analysis_id,
            context_hash=context_hash,
            algorithm_version="TRACE-RCE-v1.0",
            disease_name=disease_name,
            disease_family=disease_family,
            ranked_causes=ranked_causes,
            reasoning_chain=reasoning_chain,
            evidence_graph=evidence_graph,
            conflicts_detected=all_conflicts,
            top_intervention=top_intervention,
            risk_forecast=risk_forecast,
            modality_reliability_scores=reliability,
            causes_evaluated=len(scored_hypotheses),
            causes_above_threshold=len(ranked_causes),
            analysis_latency_ms=latency_ms,
        )

        # Store for retrieval by analysis_id
        self._analysis_store[str(analysis_id)] = result

        logger.info(
            "[TRACE-RCE] Complete: %d/%d causes above threshold, top=%s, latency=%.1fms",
            len(ranked_causes), len(scored_hypotheses),
            top_cause.cause_id if top_cause else "none",
            latency_ms,
        )

        return result

    async def explain(self, analysis_id: str) -> Dict[str, Any]:
        """
        Retrieve the full explanation for a previous analysis by its ID.

        Returns the serialized RootCauseResult as a dictionary,
        including reasoning chain, evidence graph, and confidence intervals.
        """
        result = self._analysis_store.get(analysis_id)
        if not result:
            return {
                "error": "analysis_not_found",
                "analysis_id": analysis_id,
                "message": f"No analysis found for ID '{analysis_id}'. "
                           f"The analysis may have been discarded or the ID is incorrect.",
            }

        return result.model_dump(mode="json")
