"""
Adaptive Evidence Fusion (AEF) Operator
=========================================
Computes the Causal Attribution Score (CAS) for each causal hypothesis
by fusing multi-modal evidence streams with reliability-adjusted weights.

Confidence Calibration Module (CCM)
=====================================
Applies Platt-inspired logistic calibration and estimates epistemic
and aleatoric uncertainty for each scored cause.

Conflict Resolution Layer (CRL)
=================================
Detects and resolves contradictions between modality evidence signals,
discounting CAS where evidence conflicts exceed the threshold τ.

Mathematical Reference:

    W_m(c) = (w_m × r_m) / Σ_m' (w_m' × r_m')
    CAS_raw(c) = Σ_m [W_m(c) × e_m(c)]

    conflict(m1, m2, c) = |e_m1(c) - e_m2(c)|
    CAS_final(c) = CAS_raw(c) × (1 - 0.3 × mean_conflict(c))  if conflict > τ

    P_cal(c) = σ(a × CAS_final(c) + b)   [a=5.0, b=-2.5]
    U_epistemic(c) = 1 - Σ_m [W_m(c) × r_m]
    U_aleatoric(c) = (1 - confidence) × 0.3 + (1 - is_valid_quality) × 0.2
"""

from __future__ import annotations
import math
import itertools
from typing import Dict, List, Tuple
from backend.research.root_cause_engine.ontology import CausalHypothesis


# =============================================================================
# Constants
# =============================================================================

_CONFLICT_THRESHOLD = 0.40      # τ: conflict detection threshold
_CONFLICT_DISCOUNT   = 0.30     # α_conflict: CAS discount factor
_PLATT_A = 5.0                  # calibration slope
_PLATT_B = -2.5                 # calibration bias
_MIN_THRESHOLD = 0.15           # minimum CAS_final to include a cause in results


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# =============================================================================
# Adaptive Evidence Fusion (AEF)
# =============================================================================

def compute_reliability_adjusted_weights(
    prior_weights: Dict[str, float],
    reliability: Dict[str, float],
) -> Dict[str, float]:
    """
    W_m = (w_m × r_m) / Σ_m' (w_m' × r_m')

    Returns normalized reliability-adjusted weights that sum to 1.
    """
    raw = {m: prior_weights[m] * reliability[m] for m in prior_weights}
    total = sum(raw.values())
    if total == 0.0:
        # Uniform fallback
        n = len(raw)
        return {m: 1.0 / n for m in raw}
    return {m: v / total for m, v in raw.items()}


def compute_cas_raw(
    evidence_scores: Dict[str, float],
    adjusted_weights: Dict[str, float],
) -> float:
    """
    CAS_raw(c) = Σ_m [ W_m(c) × e_m(c) ]
    """
    return sum(adjusted_weights[m] * evidence_scores[m] for m in evidence_scores)


# =============================================================================
# Conflict Resolution Layer (CRL)
# =============================================================================

def detect_conflicts(
    evidence_scores: Dict[str, float],
    cause_id: str,
) -> Tuple[List[dict], float]:
    """
    Compute pairwise conflict scores between all modality pairs.

    conflict(m1, m2, c) = |e_m1(c) - e_m2(c)|

    Returns:
        conflicts: list of conflict event dicts
        mean_conflict: mean of all pairwise conflicts (used in discounting)
    """
    modalities = list(evidence_scores.keys())
    pairs = list(itertools.combinations(modalities, 2))

    conflicts = []
    conflict_values = []

    for m1, m2 in pairs:
        conf = abs(evidence_scores[m1] - evidence_scores[m2])
        conflict_values.append(conf)
        if conf > _CONFLICT_THRESHOLD:
            conflicts.append({
                "modality_a": m1,
                "modality_b": m2,
                "cause_id": cause_id,
                "conflict_score": round(conf, 4),
            })

    mean_conf = sum(conflict_values) / len(conflict_values) if conflict_values else 0.0
    return conflicts, mean_conf


def apply_conflict_discounting(cas_raw: float, mean_conflict: float) -> Tuple[float, float]:
    """
    CAS_final(c) = CAS_raw(c) × (1 - α_conflict × mean_conflict(c))

    Returns: (cas_final, discount_applied)
    """
    if mean_conflict <= 0.0:
        return cas_raw, 0.0

    discount = _CONFLICT_DISCOUNT * mean_conflict
    cas_final = cas_raw * (1.0 - discount)
    cas_final = min(max(cas_final, 0.0), 1.0)
    return cas_final, discount


# =============================================================================
# Confidence Calibration Module (CCM)
# =============================================================================

def calibrate_confidence(
    cas_final: float,
    evidence_scores: Dict[str, float],
    adjusted_weights: Dict[str, float],
    reliability: Dict[str, float],
    vision_confidence: float,
    is_valid_image: bool,
) -> dict:
    """
    Computes calibrated confidence and uncertainty bounds.

    P_cal(c) = σ(a × CAS_final(c) + b)

    U_epistemic(c) = 1 - Σ_m [ W_m(c) × r_m ]
    U_aleatoric(c) = (1 - confidence_score) × 0.3 + (1 - is_valid_quality) × 0.2
    U_total(c) = clip(0.6 × U_epistemic + 0.4 × U_aleatoric, 0, 1)

    CI(c) = [P_cal × (1 - U_total), P_cal]

    Returns dict compatible with ConfidenceInterval schema.
    """
    # Platt calibration
    p_cal = _sigmoid(_PLATT_A * cas_final + _PLATT_B)

    # Epistemic uncertainty: how well-covered are the evidence sources?
    u_epistemic = 1.0 - sum(adjusted_weights[m] * reliability[m] for m in adjusted_weights)
    u_epistemic = min(max(u_epistemic, 0.0), 1.0)

    # Aleatoric uncertainty: sensor/measurement noise
    quality_penalty = 0.2 if not is_valid_image else 0.0
    u_aleatoric = (1.0 - vision_confidence) * 0.3 + quality_penalty
    u_aleatoric = min(max(u_aleatoric, 0.0), 1.0)

    # Total uncertainty
    u_total = 0.6 * u_epistemic + 0.4 * u_aleatoric
    u_total = min(max(u_total, 0.0), 1.0)

    lower = p_cal * (1.0 - u_total)

    return {
        "lower":                  round(lower, 4),
        "upper":                  round(p_cal, 4),
        "uncertainty_epistemic":  round(u_epistemic, 4),
        "uncertainty_aleatoric":  round(u_aleatoric, 4),
        "uncertainty_total":      round(u_total, 4),
    }
