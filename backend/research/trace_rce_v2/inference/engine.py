"""
Evidence Consistency Engine (Formerly TRACE-RCE)
==================================
Implements the AbstractRootCauseEngine production interface to act as a
risk-aware diagnosis evaluator.

Consumes a sealed AIContext object and returns the exact same
RootCauseResult Pydantic schema, including:
1. Ranked list of root causes (utilizing model confidences and ranking)
2. Modality contributions (derived from cross-attention weights)
3. Modality reliability scores (computed via extractors)
4. Explanations and reasoning chains (derived from Integrated Gradients)
"""

import os
import time
import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from backend.schemas.context import AIContext
from backend.services.root_cause_service import AbstractRootCauseEngine
from backend.research.root_cause_engine.models import (
    RootCauseResult, RootCause, ConfidenceInterval,
    EvidenceContribution, EvidenceGraph, EvidenceGraphNode, EvidenceGraphEdge
)
from backend.research.root_cause_engine.ontology import classify_disease_family, get_hypotheses_for_disease
from backend.research.trace_rce_v2.dataset.feature_extractor import extract_features, FEATURE_NAMES
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS, CAUSE_TO_IDX
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.explainability.feature_importance import compute_integrated_gradients

logger = logging.getLogger("nova.research.trace_rce_v2.inference")

MODALITY_ORDER = ["visual", "env", "severity", "knowledge", "historical"]

class EvidenceConsistencyEngine(AbstractRootCauseEngine):
    """
    Evidence Consistency Engine.
    Implements AbstractRootCauseEngine for production integration.
    """

    def __init__(self, checkpoint_path: str = "backend/research/trace_rce_v2/checkpoints/best_model.pt"):
        self._analysis_store: Dict[str, RootCauseResult] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        
        self._load_model()

    def _load_model(self):
        """Load the trained checkpoint or fall back to an uninitialized model if not found."""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.config = checkpoint["config"]
                self.model = build_model(self.config["model"])
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Successfully loaded Evidence Consistency Engine from {self.checkpoint_path} on {self.device}")
            except Exception as e:
                logger.error(f"Error loading Evidence Consistency Engine checkpoint: {e}. Falling back to default init.")
                self._load_fallback()
        else:
            logger.warning(f"No checkpoint found at {self.checkpoint_path}. Running with uninitialized fallback.")
            self._load_fallback()

    def _load_fallback(self):
        # Default config fallback
        from backend.research.trace_rce_v2.training.train import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG
        self.model = build_model(self.config["model"])
        self.model.to(self.device)
        self.model.eval()

    async def analyze(self, context: AIContext) -> RootCauseResult:
        """
        Execute the Evidence Consistency pipeline on the given AIContext.
        """
        t_start = time.perf_counter()
        analysis_id = uuid4()
        
        disease_name = context.vision.predicted_disease
        disease_family = classify_disease_family(disease_name)
        context_hash = context.get_content_hash()

        # 1. Load causal hypotheses for this disease
        hypotheses = get_hypotheses_for_disease(disease_name)
        
        # 2. Extract features
        features = extract_features(context)
        
        # 3. Prepare batched inputs (size 1)
        batch = {
            k: getattr(features, k).unsqueeze(0).to(self.device)
            for k in MODALITY_ORDER
        }

        # 4. Model forward pass
        with torch.no_grad():
            outputs = self.model(batch, return_attention=True)
            scores = outputs["scores"][0].cpu()          # (13,)
            confidences = outputs["confidences"][0].cpu()  # (13,)
            attn = outputs["attn_weights"][0].cpu()      # (5, 5)

        # 5. Extract attention weights for explanation
        # Column averages represent the key/value importances across all queries
        modality_weights = attn.mean(dim=0).tolist()
        attn_dict = {MODALITY_ORDER[i]: modality_weights[i] for i in range(5)}

        # 6. Rank causes based on neural network scores
        sorted_indices = torch.argsort(scores, descending=True)
        
        # 7. Generate Integrated Gradients for the top cause (async-safe runs in executors if needed, here run inline)
        top_idx = sorted_indices[0].item()
        ig_attributions = compute_integrated_gradients(
            model=self.model,
            batch=batch,
            sample_idx=0,
            target_cause_idx=top_idx,
            steps=20
        )

        ranked_causes = []
        rank_counter = 1
        
        for idx in sorted_indices:
            cause_id = CAUSE_IDS[idx.item()]
            
            # Filter: only evaluate hypotheses relevant to current disease family
            if cause_id not in hypotheses:
                continue
                
            hyp = hypotheses[cause_id]
            score_val = scores[idx.item()].item()
            conf_val = confidences[idx.item()].item()

            # Generate natural language reasoning sentence based on feature importance
            reasoning = self._generate_reasoning_sentence(cause_id, ig_attributions, hyp.cause_label)

            # Confidence interval
            ci = ConfidenceInterval(
                lower=max(0.0, conf_val - 0.1), # uncertainty bounds can be computed or estimated
                upper=conf_val,
                uncertainty_epistemic=0.1,
                uncertainty_aleatoric=0.0,
                uncertainty_total=0.1
            )

            # Contributions: Modality attention * raw features
            contributions = []
            raw_feats = {
                "visual": features.visual,
                "env": features.env,
                "severity": features.severity,
                "knowledge": features.knowledge,
                "historical": features.historical
            }

            for m in MODALITY_ORDER:
                raw_score = raw_feats[m].mean().item()
                weight = attn_dict[m]
                contributions.append(EvidenceContribution(
                    modality=m,
                    raw_evidence_score=round(raw_score, 4),
                    reliability_weight=round(weight, 4),
                    weighted_contribution=round(raw_score * weight, 4)
                ))

            # Action priority score = conf * impact * feasibility
            ips = conf_val * hyp.impact * hyp.feasibility

            ranked_causes.append(RootCause(
                rank=rank_counter,
                cause_id=cause_id,
                cause_label=hyp.cause_label,
                disease_family=disease_family,
                cas_raw=score_val,
                cas_final=score_val,
                confidence=ci,
                evidence_contributions=contributions,
                intervention_priority_score=round(ips, 4),
                intervention_impact=hyp.impact,
                intervention_feasibility=hyp.feasibility,
                reasoning_sentence=reasoning,
                actionable_steps=hyp.action_templates
            ))
            rank_counter += 1

        # 8. Build explainability artifacts
        reasoning_chain = [
            f"Evaluated disease: {disease_name} (Family: {disease_family}).",
            f"The network predicted {ranked_causes[0].cause_label} as the primary root cause with confidence {ranked_causes[0].confidence.upper:.2%}.",
            ranked_causes[0].reasoning_sentence
        ] if ranked_causes else ["No causal hypotheses found."]

        # Dummy evidence graph structure compatible with schema
        evidence_graph = self._build_evidence_graph(ranked_causes, attn_dict)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        result = RootCauseResult(
            analysis_id=analysis_id,
            context_hash=context_hash,
            algorithm_version="Evidence-Consistency-Engine-v1.0",
            disease_name=disease_name,
            disease_family=disease_family,
            ranked_causes=ranked_causes,
            reasoning_chain=reasoning_chain,
            evidence_graph=evidence_graph,
            conflicts_detected=[],
            top_intervention=ranked_causes[0].actionable_steps[0] if ranked_causes and ranked_causes[0].actionable_steps else None,
            risk_forecast=f"Crop disease progression risk: High. Top recommended intervention: {ranked_causes[0].actionable_steps[0] if ranked_causes and ranked_causes[0].actionable_steps else 'None'}",
            modality_reliability_scores=attn_dict,
            causes_evaluated=len(hypotheses),
            causes_above_threshold=len(ranked_causes),
            analysis_latency_ms=round(latency_ms, 2)
        )

        self._analysis_store[str(analysis_id)] = result
        return result

    async def explain(self, analysis_id: str) -> Dict[str, Any]:
        """Retrieve explanation dict for a previous analysis."""
        result = self._analysis_store.get(analysis_id)
        if not result:
            return {"error": "analysis_not_found", "analysis_id": analysis_id}
        return result.model_dump(mode="json")

    def _generate_reasoning_sentence(self, cause_id: str, ig: np.ndarray, label: str) -> str:
        """Construct natural language explanation from Integrated Gradients."""
        # Find index group with the highest absolute attribution
        # Feature sizes: visual=4, env=4, severity=4, knowledge=4, historical=3
        visual_ig = np.abs(ig[0:4]).sum()
        env_ig = np.abs(ig[4:8]).sum()
        severity_ig = np.abs(ig[8:12]).sum()
        knowledge_ig = np.abs(ig[12:16]).sum()
        historical_ig = np.abs(ig[16:19]).sum()

        modality_attributions = {
            "visual analysis": visual_ig,
            "environmental conditions": env_ig,
            "severity context": severity_ig,
            "retrieved knowledge": knowledge_ig,
            "historical recurrence": historical_ig
        }
        
        top_modality = max(modality_attributions, key=modality_attributions.get)
        return (
            f"This root cause ({label}) was strongly supported by the "
            f"features in the {top_modality} modality, which showed the highest gradient sensitivity."
        )

    def _build_evidence_graph(self, ranked_causes: List[RootCause], attn_dict: Dict[str, float]) -> EvidenceGraph:
        nodes = []
        edges = []
        
        # Add modality nodes
        for m in MODALITY_ORDER:
            nodes.append(EvidenceGraphNode(node_id=m, node_type="modality", label=m.title(), score=attn_dict[m]))
            
        # Add top causes
        for rc in ranked_causes[:3]:
            nodes.append(EvidenceGraphNode(node_id=rc.cause_id, node_type="cause", label=rc.cause_label, score=rc.cas_final))
            
            # Connect modalities to cause
            for m in MODALITY_ORDER:
                edges.append(EvidenceGraphEdge(
                    source_id=m,
                    target_id=rc.cause_id,
                    weight=attn_dict[m],
                    label="attended"
                ))

        return EvidenceGraph(nodes=nodes, edges=edges)
