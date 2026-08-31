"""
TRACE-RCE Evaluation Baselines
==============================
Phase 3.2 — Baselines

Implements 4 baseline engines for causal attribution comparison:
  1. RuleBasedBaseline: Simple domain-heuristic rules
  2. PureRAGBaseline: Text-keyword proximity scoring only
  3. LLMStyleBaseline: Simulates a large language model reasoning on classification label
  4. SimpleWeightedFusionBaseline: Static fusion without calibration/CRL/dynamic weights

All baselines implement a standard evaluate interface.
"""

from __future__ import annotations
import time
from typing import Dict, List, Any
from backend.schemas.context import AIContext
from backend.research.root_cause_engine.ontology import (
    get_hypotheses_for_disease, classify_disease_family, CAUSAL_ONTOLOGY
)
from backend.research.root_cause_engine.evidence import (
    extract_visual_evidence, extract_environmental_evidence,
    extract_concern_evidence, extract_knowledge_evidence,
    extract_historical_evidence, PRIOR_MODALITY_WEIGHTS
)


class BaseBaseline:
    """Abstract baseline class."""
    name = "Base"

    def analyze_sync(self, context: AIContext) -> List[str]:
        """Synchronously analyzes context, returning ranked cause IDs."""
        raise NotImplementedError

    async def analyze(self, context: AIContext) -> Dict[str, Any]:
        """Asynchronously runs the baseline, wrapping with performance metadata."""
        t_start = time.perf_counter()
        predicted = self.analyze_sync(context)
        latency = (time.perf_counter() - t_start) * 1000.0
        return {
            "baseline": self.name,
            "predicted_causes": predicted,
            "latency_ms": round(latency, 2)
        }


# =============================================================================
# 1. Rule-Based Baseline
# =============================================================================

class RuleBasedBaseline(BaseBaseline):
    """
    Standard expert system style rule base.
    Uses hard-coded threshold conditions to suggest root causes.
    """
    name = "Rule-Based"

    def analyze_sync(self, context: AIContext) -> List[str]:
        disease = context.vision.predicted_disease
        family = classify_disease_family(disease)
        
        temp = context.weather.temperature_7d_avg
        humidity = context.weather.humidity_7d_avg
        rain = context.weather.total_precipitation_mm
        wetness = context.weather.leaf_wetness_hours
        ph = 6.5  # default if not explicitly in custom metadata
        
        # Try to retrieve ph from custom_metadata
        if context.custom_metadata and "ph_level" in context.custom_metadata:
            ph = context.custom_metadata["ph_level"]

        predicted = []

        if family == "FUNGAL":
            if wetness >= 8.0:
                predicted.append("excessive_leaf_wetness")
            if humidity >= 78.0:
                predicted.append("high_humidity_conduciveness")
            if 20.0 <= temp <= 28.0:
                predicted.append("temperature_optimal_for_fungus")
            if context.history.total_previous_diagnoses >= 5:
                predicted.append("soil_borne_pathogen_carryover")
            if humidity >= 70.0 and wetness >= 6.0:
                predicted.append("poor_air_circulation")
                
        elif family == "BACTERIAL":
            if rain >= 15.0:
                predicted.append("rainfall_splash_dispersal")
            if temp >= 32.0:
                predicted.append("high_temperature_stress_bacterial")
            if humidity >= 60.0 and rain >= 5.0:
                predicted.append("mechanical_damage_entry_point")
                
        elif family == "VIRAL":
            if temp >= 28.0 and humidity <= 50.0:
                predicted.append("insect_vector_proliferation")
            if temp >= 32.0 and rain <= 3.0:
                predicted.append("drought_stress_susceptibility")
                
        elif family == "ABIOTIC":
            # Check soil pH stress or water stress
            if ph < 5.5 or ph > 7.5:
                predicted.append("soil_pH_imbalance")
            if rain <= 3.0:
                predicted.append("drought_stress_susceptibility")
            if rain >= 25.0:
                predicted.append("water_stress_overwatering")
            predicted.append("nutrient_deficiency")

        # fallback default causes for the family if nothing matched
        if not predicted:
            if family == "FUNGAL":
                predicted = ["excessive_leaf_wetness", "high_humidity_conduciveness"]
            elif family == "BACTERIAL":
                predicted = ["rainfall_splash_dispersal", "mechanical_damage_entry_point"]
            elif family == "VIRAL":
                predicted = ["insect_vector_proliferation", "drought_stress_susceptibility"]
            else:
                predicted = ["nutrient_deficiency", "soil_pH_imbalance"]

        return predicted


# =============================================================================
# 2. Pure RAG Baseline
# =============================================================================

class PureRAGBaseline(BaseBaseline):
    """
    RAG-only baseline.
    Looks strictly at term frequencies of cause keywords in retrieved text.
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
        "ph": "soil_pH_imbalance",
        "drainage": "water_stress_overwatering",
    }

    def analyze_sync(self, context: AIContext) -> List[str]:
        text = context.knowledge.context_text_block.lower()
        if not text:
            # Fallback to default hypotheses for disease
            hyps = get_hypotheses_for_disease(context.vision.predicted_disease)
            return list(hyps.keys())[:3]

        seen = set()
        predicted = []
        for kw, cause in self._KEYWORD_MAP.items():
            if kw in text and cause not in seen:
                predicted.append(cause)
                seen.add(cause)

        # Ensure we return at least some candidates matching the disease family
        hyps = get_hypotheses_for_disease(context.vision.predicted_disease)
        family_causes = [c for c in predicted if c in hyps]
        
        # Fill up to 3 elements if needed
        for c in hyps:
            if c not in family_causes:
                family_causes.append(c)
                if len(family_causes) >= 3:
                    break

        return family_causes[:3]


# =============================================================================
# 3. LLM-Only Reasoning Baseline
# =============================================================================

class LLMStyleBaseline(BaseBaseline):
    """
    Simulates general-knowledge LLM reasoning.
    It has access ONLY to the disease name classification and family,
    mapping them to a fixed rank based on standard published agronomic profiles,
    completely ignoring GradCAM, weather, RAG, and history.
    """
    name = "LLM-Only"

    _LLM_PRIORS = {
        "FUNGAL": ["excessive_leaf_wetness", "high_humidity_conduciveness", "temperature_optimal_for_fungus"],
        "BACTERIAL": ["mechanical_damage_entry_point", "rainfall_splash_dispersal", "high_temperature_stress_bacterial"],
        "VIRAL": ["insect_vector_proliferation", "drought_stress_susceptibility"],
        "ABIOTIC": ["nutrient_deficiency", "soil_pH_imbalance", "drought_stress_susceptibility"],
    }

    def analyze_sync(self, context: AIContext) -> List[str]:
        disease = context.vision.predicted_disease
        family = classify_disease_family(disease)
        
        # LLM returns general priors for this family
        causes = self._LLM_PRIORS.get(family, ["nutrient_deficiency"])
        
        # Verify they exist in the ontology for this disease
        hyps = get_hypotheses_for_disease(disease)
        valid_causes = [c for c in causes if c in hyps]
        
        # Fill with rest of hypotheses if we don't have enough
        for c in hyps:
            if c not in valid_causes:
                valid_causes.append(c)
                
        return valid_causes[:3]


# =============================================================================
# 4. Simple Weighted Fusion Baseline
# =============================================================================

class SimpleWeightedFusionBaseline(BaseBaseline):
    """
    Combines evidence using static prior weights.
    No dynamic reliability scaling (blurry images or missing weather don't change weights).
    No conflict resolution discounting.
    No confidence calibration.
    """
    name = "Simple-Fusion"

    def analyze_sync(self, context: AIContext) -> List[str]:
        disease = context.vision.predicted_disease
        family = classify_disease_family(disease)
        hyps = get_hypotheses_for_disease(disease)
        
        scored = []
        for cause_id, hyp in hyps.items():
            e_scores = {
                "visual":        extract_visual_evidence(context, hyp),
                "environmental": extract_environmental_evidence(context, hyp),
                "severity":      extract_concern_evidence(context, hyp),
                "knowledge":     extract_knowledge_evidence(context, hyp),
                "historical":    extract_historical_evidence(context, hyp),
            }
            
            # Static weighted average using priors
            score = sum(PRIOR_MODALITY_WEIGHTS[m] * e_scores[m] for m in PRIOR_MODALITY_WEIGHTS)
            scored.append((cause_id, score))
            
        # Rank by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]
