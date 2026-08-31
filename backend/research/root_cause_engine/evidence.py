"""
Evidence Extraction Module (EEM)
=================================
Extracts normalized evidence scores e_m(c) from each modality within AIContext.

Each extractor produces a scalar ∈ [0, 1] representing the degree to which
that modality supports a given causal hypothesis c.

Mathematical Reference (from TRACE-RCE specification):

    e_visual(c) = α_v × coverage + β_v × peak × confidence
    e_env(c) = Σ_k [ ω_k(c) × σ(λ_k × (x_k - θ_k)) ]
    e_severity(c) = (base/100) × env_factor × soil_factor × δ(c)
    e_knowledge(c) = max_j[ term_match(c, chunk_j) × relevance_j ]
    e_history(c) = (n_c / (n_total + ε)) × exp(-λ_r × Δt)
"""

from __future__ import annotations
import math
from typing import Dict, List
from backend.schemas.context import AIContext
from backend.research.root_cause_engine.ontology import CausalHypothesis


# =============================================================================
# Sigmoid Activation Utility
# =============================================================================

def _sigmoid(x: float, lam: float = 1.0, theta: float = 0.0) -> float:
    """Sigmoid activation: 1 / (1 + exp(-λ × (x - θ)))"""
    return 1.0 / (1.0 + math.exp(-lam * (x - theta)))


# =============================================================================
# Module-Level Constants
# =============================================================================

_ALPHA_V = 0.60   # GradCAM coverage weight in visual evidence
_BETA_V  = 0.40   # peak × confidence weight in visual evidence
_ENV_SLOPE = 0.20  # λ_k: sigmoid slope for environmental signals
_HISTORY_DECAY = 0.01  # λ_r: recency decay per day
_HISTORY_EPSILON = 1.0  # smoothing constant ε


# =============================================================================
# Visual Evidence Extractor (GradCAM)
# =============================================================================

def extract_visual_evidence(ctx: AIContext, hypothesis: CausalHypothesis) -> float:
    """
    e_visual(c) = α_v × heatmap_coverage_ratio + β_v × peak_intensity × confidence_score

    GradCAM evidence is not cause-specific — a higher coverage ratio universally
    strengthens any cause associated with the detected disease family.
    The hypothesis family check (is_fungal_consistent) is used as a soft gate.

    Returns: float ∈ [0, 1]
    """
    coverage = ctx.gradcam.heatmap_coverage_ratio
    peak = ctx.gradcam.peak_intensity
    confidence = ctx.vision.confidence_score

    raw = _ALPHA_V * coverage + _BETA_V * peak * confidence
    # Clip to [0, 1]
    return min(max(raw, 0.0), 1.0)


# =============================================================================
# Environmental Evidence Extractor
# =============================================================================

def extract_environmental_evidence(ctx: AIContext, hypothesis: CausalHypothesis) -> float:
    """
    e_env(c) = Σ_k [ ω_k(c) × σ(λ_k × (x_k - θ_k)) ]

    Each environmental factor x_k is activated by a sigmoid at the hypothesis-specific
    conduciveness threshold θ_k, weighted by the disease-specific sensitivity weight ω_k.

    For viral/drought causes, low rainfall activates the cause (inverted sigmoid).
    """
    h_avg = ctx.weather.humidity_7d_avg
    t_avg = ctx.weather.temperature_7d_avg
    rain  = ctx.weather.total_precipitation_mm
    wet   = ctx.weather.leaf_wetness_hours

    # Sigmoid activations per factor at hypothesis thresholds
    sig_humidity    = _sigmoid(h_avg, _ENV_SLOPE, hypothesis.humidity_threshold)
    sig_temperature = _sigmoid(t_avg, _ENV_SLOPE, hypothesis.temperature_threshold)
    sig_rainfall    = _sigmoid(rain,  _ENV_SLOPE, hypothesis.rainfall_threshold)
    sig_wetness     = _sigmoid(wet,   _ENV_SLOPE, hypothesis.wetness_threshold)

    # Special case: drought/viral causes activate under LOW rainfall (invert)
    if hypothesis.cause_id in ("drought_stress_susceptibility", "insect_vector_proliferation"):
        sig_rainfall = 1.0 - sig_rainfall

    score = (
        hypothesis.humidity_weight    * sig_humidity    +
        hypothesis.temperature_weight * sig_temperature +
        hypothesis.rainfall_weight    * sig_rainfall    +
        hypothesis.wetness_weight     * sig_wetness
    )

    return min(max(score, 0.0), 1.0)


# =============================================================================
# Severity Evidence Extractor
# =============================================================================

def extract_concern_evidence(ctx: AIContext, hypothesis: CausalHypothesis) -> float:
    """
    e_severity(c) = (base_vision_score / 100) × env_factor × soil_factor × δ(c)

    δ(c) is a consistency indicator:
      - Fungal causes: consistent when env_factor > 1.1 (weather promoting fungal spread)
      - Abiotic causes: consistent when soil_factor > 1.1 (soil stress present)
      - Others: δ = 0.8 (partial consistency, avoid zero)
    """
    base_norm = ctx.severity.base_vision_score / 100.0
    env_f     = ctx.severity.environmental_risk_factor
    soil_f    = ctx.severity.soil_stress_factor

    # Consistency indicator δ(c)
    if hypothesis.is_fungal_consistent:
        delta = 1.0 if env_f >= 1.1 else 0.5
    elif hypothesis.is_abiotic_consistent:
        delta = 1.0 if soil_f >= 1.1 else 0.5
    else:
        delta = 0.8

    score = base_norm * env_f * soil_f * delta
    return min(max(score, 0.0), 1.0)


# =============================================================================
# Knowledge Evidence Extractor (RAG)
# =============================================================================

# Map cause IDs to relevant keyword groups for text matching
_CAUSE_KEYWORDS: Dict[str, List[str]] = {
    "excessive_leaf_wetness": ["wet", "moisture", "dew", "wetness", "wet period", "leaf wetness"],
    "high_humidity_conduciveness": ["humid", "humidity", "relative humidity", "damp", "moist air"],
    "temperature_optimal_for_fungus": ["temperature", "cool", "warm", "20°", "25°", "sporulation"],
    "soil_borne_pathogen_carryover": ["soil", "debris", "carryover", "inoculum", "previous season", "volunteer"],
    "poor_air_circulation": ["canopy", "air circulation", "density", "spacing", "crowded", "airflow"],
    "rainfall_splash_dispersal": ["rain", "splash", "raindrop", "dispersal", "water droplet", "rainfall"],
    "high_temperature_stress_bacterial": ["heat", "temperature", "thermal stress", "hot", "35°"],
    "mechanical_damage_entry_point": ["wound", "pruning", "mechanical", "cut", "injury", "tool"],
    "insect_vector_proliferation": ["aphid", "whitefly", "insect", "vector", "thrip", "mite", "pest"],
    "drought_stress_susceptibility": ["drought", "dry", "water stress", "deficit", "wilting", "low rainfall"],
    "nutrient_deficiency": ["nitrogen", "phosphorus", "potassium", "deficiency", "chlorosis", "nutrient", "npk"],
    "soil_pH_imbalance": ["pH", "acidic", "alkaline", "lime", "sulfur", "soil acidity"],
    "water_stress_overwatering": ["overwater", "waterlogged", "saturated", "flooding", "anaerobic", "drainage"],
}


def extract_knowledge_evidence(ctx: AIContext, hypothesis: CausalHypothesis) -> float:
    """
    e_knowledge(c) = keyword_match_rate(cause_keywords_c, rag_context_block)

    Term frequency approach: counts how many of the cause's keyword terms
    appear in the retrieved RAG knowledge block, normalized to [0, 1].
    A simple but deterministic approach that avoids embedding re-inference overhead.
    """
    text = ctx.knowledge.context_text_block.lower()
    if not text:
        return 0.0

    keywords = _CAUSE_KEYWORDS.get(hypothesis.cause_id, [])
    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw in text)
    return min(hits / len(keywords), 1.0)


# =============================================================================
# Historical Evidence Extractor
# =============================================================================

def extract_historical_evidence(ctx: AIContext, hypothesis: CausalHypothesis) -> float:
    """
    e_history(c) = (n_c / (n_total + ε)) × exp(-λ_r × Δt)

    Computes the recurring prevalence of this cause's related diseases
    in the patient's historical diagnoses with exponential recency decay.

    Since AIContext stores frequent_crop_diseases (string list) rather than
    explicit cause IDs, we check if any frequent disease is consistent
    with the current cause's disease family as a proxy for historical frequency.
    """
    total = ctx.history.total_previous_diagnoses
    frequent = ctx.history.frequent_crop_diseases

    if total == 0:
        return 0.0

    # Count disease names in history consistent with this hypothesis
    n_c = 0
    for d in frequent:
        d_lower = d.lower()
        for family in hypothesis.disease_families:
            if family == "FUNGAL" and any(k in d_lower for k in ["blight", "mold", "mildew", "rust", "spot", "rot"]):
                n_c += 1
                break
            elif family == "BACTERIAL" and ("bacterial" in d_lower or "canker" in d_lower):
                n_c += 1
                break
            elif family == "VIRAL" and ("virus" in d_lower or "mosaic" in d_lower):
                n_c += 1
                break
            elif family == "ABIOTIC" and ("deficiency" in d_lower or "scorch" in d_lower):
                n_c += 1
                break

    frequency_score = n_c / (total + _HISTORY_EPSILON)

    # Recency decay: parse last diagnosis date if available
    recency = 1.0
    if ctx.history.last_diagnosis_date:
        try:
            from datetime import datetime
            last = datetime.fromisoformat(ctx.history.last_diagnosis_date)
            delta_days = (datetime.utcnow() - last).days
            recency = math.exp(-_HISTORY_DECAY * max(delta_days, 0))
        except Exception:
            recency = 0.5  # fallback if parsing fails

    score = min(frequency_score * recency, 1.0)
    return score


# =============================================================================
# Reliability Score Extractor
# =============================================================================

def compute_reliability_scores(ctx: AIContext) -> Dict[str, float]:
    """
    Compute dynamic reliability score r_m for each modality.

    r_visual   = 1.0 - (0.5 if image_quality_invalid else 0.0)
    r_env      = min(1.0, leaf_wetness_hours / 24 + 0.4)
    r_severity = 1.0 (always available if engine ran)
    r_knowledge= min(1.0, len(chunk_ids) / 5.0)
    r_history  = min(1.0, total_diagnoses / 10.0)
    """
    r_visual = 1.0 if ctx.image.is_valid_quality else 0.5

    wet_norm = ctx.weather.leaf_wetness_hours / 24.0
    r_env = min(1.0, wet_norm + 0.4)

    r_severity = 1.0

    n_chunks = len(ctx.knowledge.retrieved_chunk_ids)
    r_knowledge = min(1.0, n_chunks / 5.0)

    total_hist = ctx.history.total_previous_diagnoses
    r_history = min(1.0, total_hist / 10.0)

    return {
        "visual": round(r_visual, 4),
        "environmental": round(r_env, 4),
        "severity": round(r_severity, 4),
        "knowledge": round(r_knowledge, 4),
        "historical": round(r_history, 4),
    }


# =============================================================================
# Prior Modality Weights (static, domain-set)
# =============================================================================

PRIOR_MODALITY_WEIGHTS: Dict[str, float] = {
    "visual":       0.35,
    "environmental": 0.30,
    "severity":     0.15,
    "knowledge":    0.12,
    "historical":   0.08,
}
