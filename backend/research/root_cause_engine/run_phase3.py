"""
TRACE-RCE Phase 3 Full Evaluation Runner
========================================
Phase 3.8 — Full Evaluation Runner

Ties together dataset generation, baseline comparisons, ablation studies,
robustness stress testing, error analysis, and bootstrap statistical validation.
Outputs all structured results to backend/research/results/phase3_results.json.
"""

from __future__ import annotations
import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any

# Import validation dataset
from backend.research.root_cause_engine.dataset import generate_evaluation_dataset, EvalScenario

# Import baselines
from backend.research.root_cause_engine.baselines import (
    RuleBasedBaseline, PureRAGBaseline, LLMStyleBaseline, SimpleWeightedFusionBaseline
)

# Import engine
from backend.research.root_cause_engine.engine import TRACERootCauseEngine
from backend.research.root_cause_engine.models import RootCauseResult

# Import metrics and validation modules
from backend.research.root_cause_engine.metrics import compute_metrics_package
from backend.research.root_cause_engine.ablation import ablate_context, ABLATION_MODES
from backend.research.root_cause_engine.robustness import run_robustness_analysis
from backend.research.root_cause_engine.error_analysis import analyze_prediction_errors, generate_error_summary_report
from backend.research.root_cause_engine.statistics import (
    bootstrap_metric_ci, compute_cohens_d, paired_permutation_test, generate_calibration_curve_data
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nova.research.runner")


async def run_evaluation_pipeline() -> Dict[str, Any]:
    logger.info("Initializing Phase 3 Evaluation Pipeline...")
    
    # 1. Generate Dataset (300 cases)
    scenarios = generate_evaluation_dataset()
    logger.info(f"Generated {len(scenarios)} evaluation cases.")

    # 2. Instantiate systems
    trace_rce = TRACERootCauseEngine()
    rule_system = RuleBasedBaseline()
    rag_system = PureRAGBaseline()
    llm_system = LLMStyleBaseline()
    fusion_system = SimpleWeightedFusionBaseline()

    # Data stores for evaluations
    results_trace: List[RootCauseResult] = []
    preds_trace = []
    preds_rule = []
    preds_rag = []
    preds_llm = []
    preds_fusion = []

    # Latency lists
    lats_trace = []
    lats_rule = []
    lats_rag = []
    lats_llm = []
    lats_fusion = []

    # Calibration collection variables
    flat_confidences = []
    flat_outcomes = []
    
    # Auxiliary stats
    conflicts_detected = []
    completeness_scores = []
    determinism_results = []

    logger.info("Running inference across 5 systems...")
    for idx, case in enumerate(scenarios):
        ctx = case.context
        gt = case.ground_truth_causes
        
        # --- A. TRACE-RCE ---
        t0 = time.perf_counter()
        res: RootCauseResult = await trace_rce.analyze(ctx)
        dt = (time.perf_counter() - t0) * 1000.0
        
        # Force the generated latency into the object for reproducibility
        res.analysis_latency_ms = round(dt, 2)
        results_trace.append(res)
        
        trace_preds = [rc.cause_id for rc in res.ranked_causes]
        preds_trace.append(trace_preds)
        lats_trace.append(dt)

        # Double run for determinism check
        res2: RootCauseResult = await trace_rce.analyze(ctx)
        trace_preds2 = [rc.cause_id for rc in res2.ranked_causes]
        determinism_results.append(trace_preds == trace_preds2)

        # Track conflicts and completeness
        conflicts_detected.append(len(res.conflicts_detected) > 0)
        
        # completeness: how many modalities contribute score > 0.1 for top prediction
        if res.ranked_causes:
            top_rc = res.ranked_causes[0]
            comp_count = sum(1 for c in top_rc.evidence_contributions if c.raw_evidence_score > 0.1)
            completeness_scores.append(comp_count)
            
            # Calibration vectors: map calibrated confidence to true/false outcomes
            for rc in res.ranked_causes[:3]:  # look at top 3 predictions
                is_hit = 1 if rc.cause_id in gt else 0
                flat_confidences.append(rc.confidence.upper)
                flat_outcomes.append(is_hit)

        # --- B. Rule-Based ---
        rule_res = await rule_system.analyze(ctx)
        preds_rule.append(rule_res["predicted_causes"])
        lats_rule.append(rule_res["latency_ms"])

        # --- C. Pure RAG ---
        rag_res = await rag_system.analyze(ctx)
        preds_rag.append(rag_res["predicted_causes"])
        lats_rag.append(rag_res["latency_ms"])

        # --- D. LLM-Only ---
        llm_res = await llm_system.analyze(ctx)
        preds_llm.append(llm_res["predicted_causes"])
        lats_llm.append(llm_res["latency_ms"])

        # --- E. Simple Fusion ---
        fusion_res = await fusion_system.analyze(ctx)
        preds_fusion.append(fusion_res["predicted_causes"])
        lats_fusion.append(fusion_res["latency_ms"])

        if (idx + 1) % 50 == 0:
            logger.info(f"  Processed {idx + 1}/300 cases.")

    # 3. Compute Metrics for all systems
    logger.info("Computing metrics packages...")
    gts = [c.ground_truth_causes for c in scenarios]
    
    m_trace = compute_metrics_package(
        preds_trace, gts, [flat_confidences], [flat_outcomes],
        lats_trace, conflicts_detected, completeness_scores, determinism_results
    )
    m_rule = compute_metrics_package(preds_rule, gts, latencies=lats_rule)
    m_rag = compute_metrics_package(preds_rag, gts, latencies=lats_rag)
    m_llm = compute_metrics_package(preds_llm, gts, latencies=lats_llm)
    m_fusion = compute_metrics_package(preds_fusion, gts, latencies=lats_fusion)

    # 4. Run Ablation Study for TRACE-RCE
    logger.info("Running ablation study...")
    ablation_results = {}
    for mode in ABLATION_MODES:
        ab_preds = []
        for case in scenarios:
            ablated_ctx = ablate_context(case.context, mode)
            res = await trace_rce.analyze(ablated_ctx)
            ab_preds.append([rc.cause_id for rc in res.ranked_causes])
            
        pkg = compute_metrics_package(ab_preds, gts)
        ablation_results[mode] = {
            "precision_at_1": pkg["precision_at_1"],
            "precision_at_3": pkg["precision_at_3"],
            "recall_at_3": pkg["recall_at_3"]
        }

    # 5. Run Robustness Analysis
    logger.info("Running robustness analysis...")
    robustness_metrics = run_robustness_analysis(scenarios, results_trace)

    # 6. Run Error Analysis
    logger.info("Running error diagnostic checks...")
    failures = analyze_prediction_errors(scenarios, results_trace)
    error_summary = generate_error_summary_report(failures)

    # 7. Perform Statistical Validation
    logger.info("Performing bootstrap and significance calculations...")
    
    # Bootstrap CI for TRACE-RCE P@1 and Recall@3
    p1_scores = [1.0 if p[0] in gt else 0.0 for p, gt in zip(preds_trace, gts) if p]
    r3_scores = [1.0 if any(g in p[:3] for g in gt) else 0.0 for p, gt in zip(preds_trace, gts)]
    
    _, p1_lower, p1_upper = bootstrap_metric_ci(p1_scores)
    _, r3_lower, r3_upper = bootstrap_metric_ci(r3_scores)

    # Statistical significance tests vs baselines
    p1_rule_scores = [1.0 if p[0] in gt else 0.0 for p, gt in zip(preds_rule, gts) if p]
    p1_rag_scores = [1.0 if p[0] in gt else 0.0 for p, gt in zip(preds_rag, gts) if p]
    p1_llm_scores = [1.0 if p[0] in gt else 0.0 for p, gt in zip(preds_llm, gts) if p]
    p1_fusion_scores = [1.0 if p[0] in gt else 0.0 for p, gt in zip(preds_fusion, gts) if p]

    # Run paired permutation tests on P@1
    diff_rule, p_rule = paired_permutation_test(p1_scores, p1_rule_scores)
    diff_rag, p_rag = paired_permutation_test(p1_scores, p1_rag_scores)
    diff_llm, p_llm = paired_permutation_test(p1_scores, p1_llm_scores)
    diff_fusion, p_fusion = paired_permutation_test(p1_scores, p1_fusion_scores)

    # Cohen's d effect sizes vs baselines on P@1
    d_rule = compute_cohens_d(p1_scores, p1_rule_scores)
    d_rag = compute_cohens_d(p1_scores, p1_rag_scores)
    d_llm = compute_cohens_d(p1_scores, p1_llm_scores)
    d_fusion = compute_cohens_d(p1_scores, p1_fusion_scores)

    # Calibration Curve Bin coordinates
    calibration_curve = generate_calibration_curve_data(flat_confidences, flat_outcomes)

    logger.info("Assembling output report package...")
    final_output = {
        "summary": {
            "dataset_size": len(scenarios),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        },
        "systems_comparison": {
            "TRACE-RCE": m_trace,
            "Rule-Based": m_rule,
            "Pure-RAG": m_rag,
            "LLM-Only": m_llm,
            "Simple-Fusion": m_fusion
        },
        "ablation_study": ablation_results,
        "robustness_analysis": robustness_metrics,
        "error_analysis": {
            "summary": error_summary,
            "sample_failures": failures[:10]  # include first 10 for review
        },
        "statistical_validation": {
            "bootstrap_ci": {
                "precision_at_1": {"mean": m_trace["precision_at_1"], "lower": p1_lower, "upper": p1_upper},
                "recall_at_3": {"mean": m_trace["recall_at_3"], "lower": r3_lower, "upper": r3_upper}
            },
            "significance_tests_vs_baselines_pat1": {
                "Rule-Based": {"mean_difference": diff_rule, "p_value": p_rule, "cohens_d": d_rule},
                "Pure-RAG": {"mean_difference": diff_rag, "p_value": p_rag, "cohens_d": d_rag},
                "LLM-Only": {"mean_difference": diff_llm, "p_value": p_llm, "cohens_d": d_llm},
                "Simple-Fusion": {"mean_difference": diff_fusion, "p_value": p_fusion, "cohens_d": d_fusion}
            },
            "calibration_curve_bins": calibration_curve
        }
    }

    # Save to file
    out_dir = "backend/research/results"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase3_results.json")
    
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
        
    logger.info(f"Evaluation completed. Results successfully written to: {out_file}")
    return final_output


if __name__ == "__main__":
    asyncio.run(run_evaluation_pipeline())
