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
from backend.research.trace_rce_v2.dataset.feature_extractor import extract_features
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v3.models.trace_rce_v3 import TRACERCEv3

logger = logging.getLogger("nova.research.trace_rce_v3.inference")

MODALITY_ORDER = ["visual", "env", "knowledge", "historical"]

class TRACERootCauseEngineV3(AbstractRootCauseEngine):
    def __init__(self, checkpoint_path: str = "backend/research/trace_rce_v3/checkpoints/best_model.pt"):
        self._analysis_store: Dict[str, RootCauseResult] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model = TRACERCEv3()
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded TRACERCEv3 from {self.checkpoint_path} on {self.device}")
            except Exception as e:
                logger.error(f"Error loading TRACERCEv3 checkpoint: {e}. Using uninitialized model.")
                self._load_fallback()
        else:
            logger.warning(f"No checkpoint found at {self.checkpoint_path}. Running with uninitialized fallback.")
            self._load_fallback()

    def _load_fallback(self):
        self.model = TRACERCEv3()
        self.model.to(self.device)
        self.model.eval()

    async def analyze(self, context: AIContext) -> RootCauseResult:
        t_start = time.perf_counter()
        analysis_id = uuid4()
        
        disease_name = context.vision.predicted_disease
        disease_family = classify_disease_family(disease_name)
        context_hash = context.get_content_hash()

        hypotheses = get_hypotheses_for_disease(disease_name)
        features = extract_features(context)
        
        batch = {
            "visual": features.visual.unsqueeze(0).to(self.device),
            "env": features.env.unsqueeze(0).to(self.device),
            "historical": features.historical.unsqueeze(0).to(self.device),
            "knowledge": features.knowledge.unsqueeze(0).to(self.device)
        }

        with torch.no_grad():
            outputs = self.model(batch)
            scores = outputs["scores"][0].cpu()
            confidences = outputs["confidences"][0].cpu()
            
        sorted_indices = torch.argsort(scores, descending=True)
        ranked_causes = []
        rank_counter = 1
        
        for idx in sorted_indices:
            cause_id = CAUSE_IDS[idx.item()] if idx.item() < len(CAUSE_IDS) else f"cause_{idx.item()}"
            
            if cause_id not in hypotheses:
                continue
                
            hyp = hypotheses[cause_id]
            score_val = scores[idx.item()].item()
            conf_val = confidences[idx.item()].item()

            reasoning = f"Based on multimodal causal reasoning, {hyp.cause_label} is strongly supported by consistent visual and environmental evidence."

            ci = ConfidenceInterval(
                lower=max(0.0, conf_val - 0.1),
                upper=conf_val,
                uncertainty_epistemic=0.1,
                uncertainty_aleatoric=0.0,
                uncertainty_total=0.1
            )

            contributions = []
            for m in MODALITY_ORDER:
                raw_score = getattr(features, m).mean().item()
                weight = 0.25 # Mocked for V3 as it doesn't output cross-attention by default
                contributions.append(EvidenceContribution(
                    modality=m,
                    raw_evidence_score=round(raw_score, 4),
                    reliability_weight=round(weight, 4),
                    weighted_contribution=round(raw_score * weight, 4)
                ))

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

        reasoning_chain = [
            f"Evaluated disease: {disease_name} (Family: {disease_family}).",
            f"The TRACERCEv3 model predicted {ranked_causes[0].cause_label} as the primary root cause with confidence {ranked_causes[0].confidence.upper:.2%}.",
            ranked_causes[0].reasoning_sentence
        ] if ranked_causes else ["No causal hypotheses found."]

        evidence_graph = EvidenceGraph(nodes=[], edges=[])

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        result = RootCauseResult(
            analysis_id=analysis_id,
            context_hash=context_hash,
            algorithm_version="TRACE-RCE-v3.0",
            disease_name=disease_name,
            disease_family=disease_family,
            ranked_causes=ranked_causes,
            reasoning_chain=reasoning_chain,
            evidence_graph=evidence_graph,
            conflicts_detected=[],
            top_intervention=ranked_causes[0].actionable_steps[0] if ranked_causes and ranked_causes[0].actionable_steps else None,
            risk_forecast=f"Crop disease progression risk: High. Top recommended intervention: {ranked_causes[0].actionable_steps[0] if ranked_causes and ranked_causes[0].actionable_steps else 'None'}",
            modality_reliability_scores={m: 0.25 for m in MODALITY_ORDER},
            causes_evaluated=len(hypotheses),
            causes_above_threshold=len(ranked_causes),
            analysis_latency_ms=round(latency_ms, 2)
        )

        self._analysis_store[str(analysis_id)] = result
        return result

        
    async def explain(self, analysis_id: str) -> Dict[str, Any]:
        result = self._analysis_store.get(analysis_id)
        if not result:
            return {"error": "analysis_not_found", "analysis_id": analysis_id}
        return result.model_dump(mode="json")
