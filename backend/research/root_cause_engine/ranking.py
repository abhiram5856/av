"""
Causal Ranking Engine (CRE) + Intervention Priority Scheduler (IPS)
======================================================================
Takes the full set of CAS-scored hypotheses and produces:

  1. Ranked list of RootCause objects (sorted by CAS_final descending)
  2. Reasoning chain (natural language explanation of top-3 causes)
  3. Causal Evidence Graph (nodes = modalities + causes + disease, edges = contributions)
  4. Intervention priority sorted by IPS(c) = CAS_final × impact × feasibility
  5. Top-level risk forecast text
"""

from __future__ import annotations
from typing import Dict, List, Optional
from backend.research.root_cause_engine.models import (
    RootCause, EvidenceContribution, EvidenceGraph, EvidenceGraphNode, EvidenceGraphEdge,
    ConfidenceInterval
)
from backend.research.root_cause_engine.ontology import CausalHypothesis
from backend.research.root_cause_engine.fusion import _MIN_THRESHOLD


# =============================================================================
# Intervention Priority Score (IPS)
# =============================================================================

def compute_ips(cas_final: float, hypothesis: CausalHypothesis) -> float:
    """
    IPS(c) = CAS_final(c) × impact(c) × feasibility(c)

    Produces a priority score balancing how strongly a cause is attributed (CAS_final),
    how impactful fixing it would be (impact), and how feasibly the farmer can act (feasibility).
    """
    return round(cas_final * hypothesis.impact * hypothesis.feasibility, 4)


# =============================================================================
# Root Cause Assembly
# =============================================================================

def build_root_cause(
    rank: int,
    hypothesis: CausalHypothesis,
    disease_family: str,
    evidence_scores: Dict[str, float],
    adjusted_weights: Dict[str, float],
    cas_raw: float,
    cas_final: float,
    conflict_discount: float,
    confidence_dict: dict,
) -> RootCause:
    """
    Assembles a complete RootCause Pydantic object from scoring results.
    """
    ips = compute_ips(cas_final, hypothesis)

    # Build evidence contributions list
    contributions = []
    for modality, score in evidence_scores.items():
        w = adjusted_weights.get(modality, 0.0)
        contributions.append(EvidenceContribution(
            modality=modality,
            raw_evidence_score=round(score, 4),
            reliability_weight=round(w, 4),
            weighted_contribution=round(w * score, 4),
        ))

    # Sort contributions by weighted contribution descending (most to least impactful)
    contributions.sort(key=lambda x: x.weighted_contribution, reverse=True)

    # Build reasoning sentence
    top_contrib = contributions[0] if contributions else None
    if top_contrib:
        reason = (
            f"Attributed with CAS={cas_final:.2f}. "
            f"Primary evidence from '{top_contrib.modality}' modality "
            f"(contribution={top_contrib.weighted_contribution:.2f}). "
        )
        if conflict_discount > 0.0:
            reason += f"Note: CAS was discounted by {conflict_discount:.1%} due to cross-modal conflict. "
    else:
        reason = f"Attributed with CAS={cas_final:.2f}."

    return RootCause(
        rank=rank,
        cause_id=hypothesis.cause_id,
        cause_label=hypothesis.cause_label,
        disease_family=disease_family,
        cas_raw=round(cas_raw, 4),
        cas_final=round(cas_final, 4),
        conflict_discounting_applied=round(conflict_discount, 4),
        confidence=ConfidenceInterval(**confidence_dict),
        evidence_contributions=contributions,
        intervention_priority_score=ips,
        intervention_impact=hypothesis.impact,
        intervention_feasibility=hypothesis.feasibility,
        reasoning_sentence=reason,
        actionable_steps=hypothesis.action_templates,
    )


# =============================================================================
# Evidence Graph Builder
# =============================================================================

def build_evidence_graph(
    disease_name: str,
    disease_family: str,
    ranked_causes: List[RootCause],
    evidence_scores_per_cause: Dict[str, Dict[str, float]],
    adjusted_weights: Dict[str, float],
) -> EvidenceGraph:
    """
    Constructs a directed evidence graph:
        Modalities → (CausalHypotheses) → Disease

    Nodes:
        - One node per active modality (visual, environmental, severity, knowledge, historical)
        - One node per ranked cause
        - One disease node (root of the graph)

    Edges:
        - Modality → Cause with weight = W_m × e_m(c)
        - Cause → Disease with weight = CAS_final
    """
    nodes = []
    edges = []

    # Disease node
    disease_node = EvidenceGraphNode(
        node_id="disease",
        node_type="disease",
        label=disease_name.replace("_", " ").title(),
        score=1.0,
    )
    nodes.append(disease_node)

    # Modality nodes
    modalities = ["visual", "environmental", "severity", "knowledge", "historical"]
    for m in modalities:
        nodes.append(EvidenceGraphNode(
            node_id=f"mod_{m}",
            node_type="modality",
            label=m.capitalize(),
            score=adjusted_weights.get(m, 0.0),
        ))

    # Cause nodes + edges
    for rc in ranked_causes:
        cause_node_id = f"cause_{rc.cause_id}"
        nodes.append(EvidenceGraphNode(
            node_id=cause_node_id,
            node_type="cause",
            label=rc.cause_label,
            score=rc.cas_final,
        ))

        # Cause → Disease edge
        edges.append(EvidenceGraphEdge(
            source_id=cause_node_id,
            target_id="disease",
            weight=rc.cas_final,
            label=f"CAS={rc.cas_final:.2f}",
        ))

        # Modality → Cause edges
        e_scores = evidence_scores_per_cause.get(rc.cause_id, {})
        for m in modalities:
            contribution = adjusted_weights.get(m, 0.0) * e_scores.get(m, 0.0)
            if contribution > 0.01:  # prune negligible edges
                edges.append(EvidenceGraphEdge(
                    source_id=f"mod_{m}",
                    target_id=cause_node_id,
                    weight=round(contribution, 4),
                    label=f"w={contribution:.2f}",
                ))

    return EvidenceGraph(nodes=nodes, edges=edges)


# =============================================================================
# Reasoning Chain Builder
# =============================================================================

def build_reasoning_chain(
    disease_name: str,
    disease_family: str,
    ranked_causes: List[RootCause],
) -> List[str]:
    """
    Produces a step-by-step causal reasoning narrative for the top-3 ranked causes.
    Each sentence explains how TRACE-RCE arrived at the attribution decision.
    """
    chain = []
    display_name = disease_name.replace("_", " ").title()

    chain.append(
        f"Step 1 — Disease Classification: NOVA detected '{display_name}' "
        f"(disease family: {disease_family}). TRACE-RCE loaded the causal "
        f"hypothesis set for this disease family."
    )
    chain.append(
        "Step 2 — Multi-Modal Evidence Extraction: Visual (GradCAM), "
        "Environmental (weather), Severity, Knowledge (RAG), and Historical "
        "evidence were extracted from the AIContext."
    )
    chain.append(
        "Step 3 — Reliability Adjustment: Modality weights were scaled by "
        "data quality (image quality, weather completeness, RAG chunk count, "
        "historical depth) to produce reliability-adjusted fusion weights."
    )

    for i, rc in enumerate(ranked_causes[:3], start=4):
        top_mod = max(rc.evidence_contributions, key=lambda x: x.weighted_contribution) \
                  if rc.evidence_contributions else None
        mod_str = f"'{top_mod.modality}' (contribution={top_mod.weighted_contribution:.2f})" \
                  if top_mod else "unknown"
        chain.append(
            f"Step {i} — Cause #{rc.rank}: '{rc.cause_label}' scored "
            f"CAS_final={rc.cas_final:.2f} "
            f"(calibrated confidence: {rc.confidence.lower:.1%}–{rc.confidence.upper:.1%}). "
            f"Primary driver: {mod_str}. "
            f"Intervention Priority Score: {rc.intervention_priority_score:.2f}."
        )

    if ranked_causes:
        top = ranked_causes[0]
        chain.append(
            f"Step {len(ranked_causes[:3]) + 4} — Recommendation: '{top.cause_label}' "
            f"is identified as the primary root cause. "
            f"Immediate actions: {' | '.join(top.actionable_steps[:2])}"
        )

    return chain


# =============================================================================
# Risk Forecast Generator
# =============================================================================

def generate_risk_forecast(
    disease_family: str,
    final_severity: float,
    top_cause: Optional[RootCause],
    humidity: float,
    temperature: float,
) -> str:
    """
    Generates a short-term disease progression risk assessment text.
    """
    if not top_cause:
        return "Insufficient evidence to generate risk forecast."

    severity_label = (
        "severe" if final_severity >= 70
        else "moderate" if final_severity >= 30
        else "mild"
    )

    if disease_family == "FUNGAL":
        if humidity > 80 and 18 <= temperature <= 30:
            progression = "HIGH — ambient conditions remain conducive to rapid fungal spread."
        elif humidity > 65:
            progression = "MEDIUM — humidity levels may sustain pathogen activity."
        else:
            progression = "LOW — environmental conditions are becoming less conducive."
    elif disease_family == "BACTERIAL":
        progression = "MEDIUM — bacterial infections can spread rapidly during wet weather." \
                      if humidity > 70 else "LOW — dry conditions limit bacterial dispersal."
    elif disease_family == "VIRAL":
        progression = "HIGH — insect vectors may be active under current temperature conditions." \
                      if temperature > 28 else "MEDIUM — vector pressure is moderate."
    else:
        progression = "LOW — abiotic stress conditions require long-term soil/water management."

    return (
        f"Current severity: {severity_label.upper()} ({final_severity:.0f}/100). "
        f"Short-term progression risk: {progression} "
        f"Primary controllable factor: '{top_cause.cause_label}'."
    )
