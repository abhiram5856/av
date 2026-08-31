"""
TRACE-RCE Evaluation Harness & Ablation Study Framework
=========================================================
Phase 2.5 — Evaluation Module

Implements:
    1. Baseline comparisons (Rule-Based, Pure RAG, Pure LLM-style)
    2. TRACE-RCE evaluation metrics
    3. Ablation study: systematic modality removal
    4. Ground truth fixtures for representative test cases

Metrics:
    - Explanation Consistency (determinism check)
    - Cause Precision@K (vs agronomic ground truth)
    - Confidence Calibration Error (ECE)
    - Explanation Latency (wall-clock ms)
    - Evidence Completeness (modalities contributing > 0)
    - Conflict Detection Rate (% runs where CRL activated)
"""

from __future__ import annotations
import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from backend.schemas.context import (
    AIContext, SystemMetadataContext, UserMetadataContext, ImageMetadataContext,
    VisionDetectionContext, GradCAMContext, ConcernContext, WeatherContext,
    RAGKnowledgeContext, PatientHistoryContext
)
from backend.research.root_cause_engine.models import RootCauseResult
from backend.research.root_cause_engine.engine import TRACERootCauseEngine

logger = logging.getLogger("nova.research.evaluation")


# =============================================================================
# Ground Truth Fixtures
# =============================================================================

@dataclass
class EvalCase:
    """A single evaluation test case."""
    case_id: str
    description: str
    context: AIContext
    ground_truth_causes: List[str]  # expected top cause IDs (from agronomic literature)
    expected_family: str


def _make_context(
    disease: str, confidence: float,
    coverage: float, peak: float,
    humidity: float, temp: float, rainfall: float, wetness: float,
    env_factor: float, soil_factor: float, severity: float,
    chunk_ids: List[str], rag_text: str,
    total_hist: int, frequent: List[str], last_date: Optional[str] = None,
    blur_score: float = 85.0, is_valid: bool = True,
) -> AIContext:
    """Helper to construct an AIContext for evaluation purposes."""
    return AIContext(
        system=SystemMetadataContext(request_id="eval-001"),
        user=UserMetadataContext(user_id="eval", role="researcher", region="test", preferred_language="en"),
        image=ImageMetadataContext(
            filename="test.jpg", width=224, height=224, format="JPEG",
            aspect_ratio=1.0, blur_score=blur_score, is_valid_quality=is_valid
        ),
        vision=VisionDetectionContext(
            predicted_disease=disease, confidence_score=confidence,
            topk_predictions={disease: confidence}, concept_activations={}
        ),
        gradcam=GradCAMContext(
            heatmap_coverage_ratio=coverage, peak_intensity=peak,
            target_layer_name="features.13", s3_heatmap_url=""
        ),
        severity=ConcernContext(
            base_vision_score=severity * 0.6,
            environmental_risk_factor=env_factor,
            soil_stress_factor=soil_factor,
            concern_score=severity,
            concern_level="High" if severity >= 70 else "Medium" if severity >= 30 else "Low"
        ),
        weather=WeatherContext(
            latitude=17.38, longitude=78.48,
            temperature_7d_avg=temp, humidity_7d_avg=humidity,
            total_precipitation_mm=rainfall, leaf_wetness_hours=wetness,
            raw_forecast_summary={}
        ),
        knowledge=RAGKnowledgeContext(
            retrieved_chunk_ids=chunk_ids,
            document_sources=["plant_disease_kb.txt"],
            context_text_block=rag_text
        ),
        history=PatientHistoryContext(
            total_previous_diagnoses=total_hist,
            frequent_crop_diseases=frequent,
            last_diagnosis_date=last_date
        )
    )


# Ground truth test cases grounded in agricultural literature
EVAL_CASES: List[EvalCase] = [

    EvalCase(
        case_id="TC-01",
        description="Tomato Late Blight — High humidity, optimal fungal temperature, significant lesion area",
        context=_make_context(
            disease="tomato_late_blight", confidence=0.95,
            coverage=0.55, peak=0.88,
            humidity=87.0, temp=22.0, rainfall=18.0, wetness=14.0,
            env_factor=1.40, soil_factor=1.10, severity=72.5,
            chunk_ids=["c1", "c2", "c3", "c4"],
            rag_text="Late blight thrives in high humidity conditions above 80%. "
                     "Excessive leaf wetness duration beyond 8 hours per day dramatically increases sporulation. "
                     "Fungal pathogens like Phytophthora infestans spread rapidly via wet leaf surfaces.",
            total_hist=5, frequent=["tomato_late_blight", "early_blight"],
        ),
        ground_truth_causes=["excessive_leaf_wetness", "high_humidity_conduciveness"],
        expected_family="FUNGAL"
    ),

    EvalCase(
        case_id="TC-02",
        description="Corn Northern Leaf Blight — Moderate conditions, low RAG retrieval",
        context=_make_context(
            disease="corn_northern_leaf_blight", confidence=0.82,
            coverage=0.30, peak=0.65,
            humidity=68.0, temp=24.0, rainfall=8.0, wetness=5.0,
            env_factor=1.15, soil_factor=1.05, severity=35.0,
            chunk_ids=["c1", "c2"],
            rag_text="Northern leaf blight is favored by moderate humidity and temperatures. "
                     "Air circulation plays a key role in reducing moisture accumulation.",
            total_hist=2, frequent=["corn_northern_leaf_blight"],
        ),
        ground_truth_causes=["high_humidity_conduciveness", "temperature_optimal_for_fungus"],
        expected_family="FUNGAL"
    ),

    EvalCase(
        case_id="TC-03",
        description="Tomato Bacterial Spot — Rainfall splash dispersal scenario",
        context=_make_context(
            disease="tomato_bacterial_spot", confidence=0.78,
            coverage=0.22, peak=0.50,
            humidity=75.0, temp=30.0, rainfall=22.0, wetness=6.0,
            env_factor=1.10, soil_factor=1.0, severity=42.0,
            chunk_ids=["c1", "c2", "c3"],
            rag_text="Bacterial spot spreads rapidly via rain splash. "
                     "Mechanical damage from tools creates entry points for Xanthomonas. "
                     "High rainfall events above 15mm drive bacterial dispersal between plants.",
            total_hist=3, frequent=["bacterial_spot"],
        ),
        ground_truth_causes=["rainfall_splash_dispersal", "mechanical_damage_entry_point"],
        expected_family="BACTERIAL"
    ),

    EvalCase(
        case_id="TC-04",
        description="Tomato Yellow Leaf Curl Virus — Dry heat, vector pressure scenario",
        context=_make_context(
            disease="tomato_yellow_leaf_curl_virus", confidence=0.88,
            coverage=0.18, peak=0.40,
            humidity=38.0, temp=34.0, rainfall=2.0, wetness=1.5,
            env_factor=1.05, soil_factor=1.0, severity=28.0,
            chunk_ids=["c1", "c2", "c3"],
            rag_text="Yellow leaf curl virus is transmitted by whitefly (Bemisia tabaci) vectors. "
                     "Drought stress makes plants more susceptible to insect vectors. "
                     "High temperatures above 30°C increase vector reproduction rates.",
            total_hist=4, frequent=["yellow_leaf_curl_virus", "mosaic_virus"],
        ),
        ground_truth_causes=["insect_vector_proliferation", "drought_stress_susceptibility"],
        expected_family="VIRAL"
    ),

    EvalCase(
        case_id="TC-05",
        description="Blurry image, poor quality — reduced visual modality reliability",
        context=_make_context(
            disease="tomato_late_blight", confidence=0.55,
            coverage=0.20, peak=0.35,
            humidity=80.0, temp=23.0, rainfall=12.0, wetness=10.0,
            env_factor=1.25, soil_factor=1.0, severity=40.0,
            chunk_ids=["c1"],
            rag_text="Fungal diseases are strongly correlated with extended leaf wetness periods.",
            total_hist=0, frequent=[],
            blur_score=12.0, is_valid=False,  # blurry image
        ),
        ground_truth_causes=["excessive_leaf_wetness", "high_humidity_conduciveness"],
        expected_family="FUNGAL"
    ),
]


# =============================================================================
# Baseline Engines
# =============================================================================

class RuleBasedBaseline:
    """
    Simple rule-based baseline: IF humidity > 80 AND disease is fungal THEN cause = excessive_leaf_wetness.
    Demonstrates the limitation of hard-coded thresholds without multi-modal fusion.
    """
    name = "Rule-Based"

    async def analyze(self, context: AIContext, ground_truth: List[str]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        disease = context.vision.predicted_disease
        family = "FUNGAL" if any(k in disease for k in ["blight", "mold", "spot", "rust"]) else "OTHER"

        predicted = []
        if family == "FUNGAL":
            if context.weather.humidity_7d_avg > 80:
                predicted.append("excessive_leaf_wetness")
            if context.weather.humidity_7d_avg > 75:
                predicted.append("high_humidity_conduciveness")
            if 20 <= context.weather.temperature_7d_avg <= 30:
                predicted.append("temperature_optimal_for_fungus")

        latency = (time.perf_counter() - t_start) * 1000
        p_at_1 = 1.0 if (predicted and predicted[0] in ground_truth) else 0.0
        p_at_2 = sum(1 for p in predicted[:2] if p in ground_truth) / min(2, max(len(predicted), 1))
        return {
            "baseline": self.name,
            "predicted_causes": predicted,
            "precision_at_1": p_at_1,
            "precision_at_2": round(p_at_2, 3),
            "latency_ms": round(latency, 2),
        }


class PureRAGBaseline:
    """
    Pure RAG baseline: retrieves relevant text, searches for cause keywords.
    No numerical modality fusion.
    """
    name = "Pure-RAG"
    _KEYWORD_MAP = {
        "wetness": "excessive_leaf_wetness",
        "wet": "excessive_leaf_wetness",
        "humidity": "high_humidity_conduciveness",
        "humid": "high_humidity_conduciveness",
        "temperature": "temperature_optimal_for_fungus",
        "cool": "temperature_optimal_for_fungus",
        "rain": "rainfall_splash_dispersal",
        "splash": "rainfall_splash_dispersal",
        "aphid": "insect_vector_proliferation",
        "whitefly": "insect_vector_proliferation",
        "drought": "drought_stress_susceptibility",
        "dry": "drought_stress_susceptibility",
        "wound": "mechanical_damage_entry_point",
        "soil": "soil_borne_pathogen_carryover",
        "nutrient": "nutrient_deficiency",
    }

    async def analyze(self, context: AIContext, ground_truth: List[str]) -> Dict[str, Any]:
        t_start = time.perf_counter()
        text = context.knowledge.context_text_block.lower()
        seen, predicted = set(), []
        for kw, cause in self._KEYWORD_MAP.items():
            if kw in text and cause not in seen:
                predicted.append(cause)
                seen.add(cause)

        latency = (time.perf_counter() - t_start) * 1000
        p_at_1 = 1.0 if (predicted and predicted[0] in ground_truth) else 0.0
        p_at_2 = sum(1 for p in predicted[:2] if p in ground_truth) / min(2, max(len(predicted), 1))
        return {
            "baseline": self.name,
            "predicted_causes": predicted[:3],
            "precision_at_1": p_at_1,
            "precision_at_2": round(p_at_2, 3),
            "latency_ms": round(latency, 2),
        }


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_precision_at_k(predicted_causes: List[str], ground_truth: List[str], k: int) -> float:
    """Precision@K: fraction of top-K predicted causes that appear in ground truth."""
    if not predicted_causes:
        return 0.0
    top_k = predicted_causes[:k]
    hits = sum(1 for c in top_k if c in ground_truth)
    return hits / k


def compute_consistency(result_a: RootCauseResult, result_b: RootCauseResult) -> bool:
    """
    Explanation Consistency: check that two runs on identical AIContext produce identical results.
    TRACE-RCE is deterministic, so this should always return True.
    """
    causes_a = [rc.cause_id for rc in result_a.ranked_causes]
    causes_b = [rc.cause_id for rc in result_b.ranked_causes]
    return causes_a == causes_b and result_a.context_hash == result_b.context_hash


def compute_ece(
    results: List[RootCauseResult],
    ground_truths: List[List[str]],
    n_bins: int = 5
) -> float:
    """
    Expected Calibration Error (ECE).
    Bins predicted confidence values and measures |mean_conf - fraction_correct| per bin.
    """
    all_pairs = []
    for result, gt in zip(results, ground_truths):
        for rc in result.ranked_causes:
            is_correct = 1.0 if rc.cause_id in gt else 0.0
            all_pairs.append((rc.confidence.upper, is_correct))

    if not all_pairs:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, correct in all_pairs:
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    total = len(all_pairs)
    for b in bins:
        if b:
            avg_conf = sum(c for c, _ in b) / len(b)
            avg_acc  = sum(a for _, a in b) / len(b)
            ece += (len(b) / total) * abs(avg_conf - avg_acc)

    return round(ece, 4)


# =============================================================================
# Ablation Study
# =============================================================================

ABLATION_MODES = [
    "full",
    "ablate_visual",
    "ablate_environmental",
    "ablate_severity",
    "ablate_knowledge",
    "ablate_historical",
]


def _ablate_context(context: AIContext, mode: str) -> AIContext:
    """
    Returns a modified AIContext with a specified modality zeroed out.
    Used for ablation experiments.
    """
    if mode == "full":
        return context

    # Create mutable copies of each sub-context field
    gradcam = context.gradcam
    weather = context.weather
    severity = context.concern
    knowledge = context.knowledge
    history = context.history

    if mode == "ablate_visual":
        gradcam = GradCAMContext(
            heatmap_coverage_ratio=0.0, peak_intensity=0.0,
            target_layer_name=context.gradcam.target_layer_name,
            s3_heatmap_url=""
        )
    elif mode == "ablate_environmental":
        weather = WeatherContext(
            latitude=context.weather.latitude, longitude=context.weather.longitude,
            temperature_7d_avg=25.0, humidity_7d_avg=50.0,
            total_precipitation_mm=0.0, leaf_wetness_hours=0.0,
            raw_forecast_summary={}
        )
    elif mode == "ablate_severity":
        severity = ConcernContext(
            base_vision_score=0.0, environmental_risk_factor=1.0,
            soil_stress_factor=1.0, concern_score=0.0, concern_level="Low"
        )
    elif mode == "ablate_knowledge":
        knowledge = RAGKnowledgeContext(
            retrieved_chunk_ids=[], document_sources=[], context_text_block=""
        )
    elif mode == "ablate_historical":
        history = PatientHistoryContext(
            total_previous_diagnoses=0, frequent_crop_diseases=[]
        )

    return AIContext(
        system=context.system, user=context.user, image=context.image,
        vision=context.vision, gradcam=gradcam, severity=severity,
        weather=weather, knowledge=knowledge, history=history,
        custom_metadata=context.custom_metadata
    )


# =============================================================================
# Full Evaluation Runner
# =============================================================================

async def run_full_evaluation() -> Dict[str, Any]:
    """
    Execute the complete evaluation:
    1. TRACE-RCE full runs on all test cases
    2. Baseline comparisons
    3. Consistency check (double-run)
    4. ECE computation
    5. Ablation study (all modes × all cases)
    """
    engine = TRACERootCauseEngine()
    rule_baseline = RuleBasedBaseline()
    rag_baseline = PureRAGBaseline()

    report = {
        "algorithm": "TRACE-RCE-v1.0",
        "num_test_cases": len(EVAL_CASES),
        "trace_results": [],
        "baseline_comparison": [],
        "consistency_check": [],
        "ablation_study": [],
        "summary_metrics": {},
    }

    all_rce_results = []
    all_ground_truths = []

    # ── Main TRACE-RCE evaluation ─────────────────────────────────────────────
    for case in EVAL_CASES:
        logger.info("Evaluating case: %s", case.case_id)

        result: RootCauseResult = await engine.analyze(case.context)
        predicted = [rc.cause_id for rc in result.ranked_causes]

        p_at_1 = compute_precision_at_k(predicted, case.ground_truth_causes, k=1)
        p_at_2 = compute_precision_at_k(predicted, case.ground_truth_causes, k=2)

        all_rce_results.append(result)
        all_ground_truths.append(case.ground_truth_causes)

        # Consistency: run again and compare
        result2: RootCauseResult = await engine.analyze(case.context)
        consistent = compute_consistency(result, result2)

        report["trace_results"].append({
            "case_id": case.case_id,
            "description": case.description,
            "predicted_top_causes": predicted[:3],
            "ground_truth": case.ground_truth_causes,
            "precision_at_1": p_at_1,
            "precision_at_2": round(p_at_2, 3),
            "top_cause": result.ranked_causes[0].cause_id if result.ranked_causes else "none",
            "top_cas_final": result.ranked_causes[0].cas_final if result.ranked_causes else 0.0,
            "conflicts_detected": len(result.conflicts_detected),
            "latency_ms": result.analysis_latency_ms,
            "consistent": consistent,
        })

        report["consistency_check"].append({
            "case_id": case.case_id,
            "consistent": consistent,
            "ctx_hash_match": result.context_hash == result2.context_hash,
        })

        # Baselines
        rb = await rule_baseline.analyze(case.context, case.ground_truth_causes)
        rag = await rag_baseline.analyze(case.context, case.ground_truth_causes)
        report["baseline_comparison"].append({
            "case_id": case.case_id,
            "trace_rce": {"precision_at_1": p_at_1, "precision_at_2": round(p_at_2, 3)},
            "rule_based": rb,
            "pure_rag": rag,
        })

    # ── ECE ──────────────────────────────────────────────────────────────────
    ece = compute_ece(all_rce_results, all_ground_truths)

    # ── Ablation Study ────────────────────────────────────────────────────────
    for mode in ABLATION_MODES:
        mode_precision = []
        for case in EVAL_CASES:
            ablated_ctx = _ablate_context(case.context, mode)
            ablated_result = await engine.analyze(ablated_ctx)
            predicted = [rc.cause_id for rc in ablated_result.ranked_causes]
            p1 = compute_precision_at_k(predicted, case.ground_truth_causes, k=1)
            mode_precision.append(p1)

        avg_p1 = round(sum(mode_precision) / len(mode_precision), 3) if mode_precision else 0.0
        report["ablation_study"].append({
            "mode": mode,
            "per_case_precision_at_1": mode_precision,
            "avg_precision_at_1": avg_p1,
        })

    # ── Summary Metrics ───────────────────────────────────────────────────────
    all_p1 = [r["precision_at_1"] for r in report["trace_results"]]
    all_p2 = [r["precision_at_2"] for r in report["trace_results"]]
    all_latencies = [r["latency_ms"] for r in report["trace_results"] if r["latency_ms"]]
    conflict_rate = sum(1 for r in report["trace_results"] if r["conflicts_detected"] > 0) / len(EVAL_CASES)
    consistency_rate = sum(1 for c in report["consistency_check"] if c["consistent"]) / len(EVAL_CASES)

    report["summary_metrics"] = {
        "trace_rce_avg_precision_at_1": round(sum(all_p1) / len(all_p1), 3),
        "trace_rce_avg_precision_at_2": round(sum(all_p2) / len(all_p2), 3),
        "rule_based_avg_precision_at_1": round(
            sum(r["rule_based"]["precision_at_1"] for r in report["baseline_comparison"]) / len(EVAL_CASES), 3
        ),
        "pure_rag_avg_precision_at_1": round(
            sum(r["pure_rag"]["precision_at_1"] for r in report["baseline_comparison"]) / len(EVAL_CASES), 3
        ),
        "expected_calibration_error": ece,
        "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else 0,
        "conflict_detection_rate": round(conflict_rate, 3),
        "consistency_rate": round(consistency_rate, 3),
    }

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_full_evaluation())
    print(json.dumps(results, indent=2, default=str))
