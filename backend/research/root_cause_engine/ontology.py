"""
Causal Hypothesis Set (CHS) Ontology
=====================================
Defines the domain-grounded causal hypotheses for each disease family.
Each entry encodes:
  - cause_id:           machine-readable identifier
  - cause_label:        human-readable label
  - disease_families:   which disease families this cause applies to
  - env_weights:        environmental factor sensitivity weights ω_k(c)
  - conduciveness_thresholds: θ_k values for sigmoid activation
  - impact:             severity reduction impact if addressed (0-1)
  - feasibility:        ease of farmer intervention (0-1)
  - action_templates:   concrete actionable steps

This ontology is the domain knowledge backbone of TRACE-RCE.
It can be extended by agricultural domain experts without changing algorithm code.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CausalHypothesis:
    """
    A single causal hypothesis entry in the ontology.
    """
    cause_id: str
    cause_label: str
    disease_families: List[str]  # FUNGAL | BACTERIAL | VIRAL | ABIOTIC

    # Environmental factor weights ω_k(c) — how strongly each env var drives this cause
    humidity_weight: float = 0.0
    temperature_weight: float = 0.0
    rainfall_weight: float = 0.0
    wetness_weight: float = 0.0

    # Conduciveness thresholds θ_k — where sigmoid starts activating
    humidity_threshold: float = 70.0     # %
    temperature_threshold: float = 25.0  # °C (for fungal)
    rainfall_threshold: float = 10.0     # mm/week
    wetness_threshold: float = 8.0       # hours/day

    # Severity-family consistency: True if high severity env_factor is consistent with this cause
    is_fungal_consistent: bool = False
    is_abiotic_consistent: bool = False

    # Intervention metadata
    impact: float = 0.5          # expected severity reduction (0-1)
    feasibility: float = 0.5     # ease of farmer intervention (0-1)
    action_templates: List[str] = field(default_factory=list)


# =============================================================================
# Master Causal Hypothesis Ontology
# =============================================================================

CAUSAL_ONTOLOGY: Dict[str, CausalHypothesis] = {

    # ─────────────────────────────────────────────
    # FUNGAL DISEASE CAUSES
    # ─────────────────────────────────────────────

    "excessive_leaf_wetness": CausalHypothesis(
        cause_id="excessive_leaf_wetness",
        cause_label="Excessive Leaf Wetness Duration",
        disease_families=["FUNGAL"],
        humidity_weight=0.25,
        wetness_weight=0.60,
        rainfall_weight=0.15,
        humidity_threshold=75.0,
        wetness_threshold=8.0,
        rainfall_threshold=12.0,
        is_fungal_consistent=True,
        impact=0.80,
        feasibility=0.70,
        action_templates=[
            "Improve field drainage to reduce standing water on leaves.",
            "Avoid overhead irrigation — switch to drip irrigation.",
            "Apply protective fungicide early morning before heavy dew periods.",
            "Prune excess foliage to improve air circulation and reduce wetness duration.",
        ]
    ),

    "high_humidity_conduciveness": CausalHypothesis(
        cause_id="high_humidity_conduciveness",
        cause_label="High Ambient Humidity Conducive to Fungal Growth",
        disease_families=["FUNGAL"],
        humidity_weight=0.70,
        temperature_weight=0.20,
        wetness_weight=0.10,
        humidity_threshold=78.0,
        temperature_threshold=22.0,
        is_fungal_consistent=True,
        impact=0.60,
        feasibility=0.40,  # humidity is harder to control
        action_templates=[
            "Apply systemic fungicide (e.g., mancozeb or chlorothalonil) preventively.",
            "Use resistant variety for next crop cycle.",
            "Monitor humidity daily; apply fungicide when 3+ consecutive high-humidity days occur.",
        ]
    ),

    "temperature_optimal_for_fungus": CausalHypothesis(
        cause_id="temperature_optimal_for_fungus",
        cause_label="Temperature in Optimal Range for Fungal Sporulation (20–28°C)",
        disease_families=["FUNGAL"],
        temperature_weight=0.70,
        humidity_weight=0.30,
        temperature_threshold=24.0,
        humidity_threshold=65.0,
        is_fungal_consistent=True,
        impact=0.50,
        feasibility=0.20,  # temperature not controllable in open field
        action_templates=[
            "Apply contact fungicide during cool morning hours when sporulation peaks.",
            "Use weather-based spray scheduling to optimize fungicide timing.",
        ]
    ),

    "soil_borne_pathogen_carryover": CausalHypothesis(
        cause_id="soil_borne_pathogen_carryover",
        cause_label="Soil-Borne Pathogen Carryover from Previous Season",
        disease_families=["FUNGAL", "BACTERIAL"],
        humidity_weight=0.20,
        rainfall_weight=0.30,
        humidity_threshold=60.0,
        rainfall_threshold=8.0,
        is_fungal_consistent=True,
        impact=0.75,
        feasibility=0.60,
        action_templates=[
            "Deep-plow to bury infected crop debris below 15 cm.",
            "Apply soil fumigant or bio-fungicide (Trichoderma spp.) before next planting.",
            "Implement 3-year crop rotation with non-host species.",
        ]
    ),

    "poor_air_circulation": CausalHypothesis(
        cause_id="poor_air_circulation",
        cause_label="Poor Canopy Air Circulation Trapping Moisture",
        disease_families=["FUNGAL"],
        humidity_weight=0.50,
        wetness_weight=0.50,
        humidity_threshold=70.0,
        wetness_threshold=6.0,
        is_fungal_consistent=True,
        impact=0.65,
        feasibility=0.80,
        action_templates=[
            "Prune inner branches to open the canopy.",
            "Widen plant spacing in the next sowing cycle.",
            "Train plants on stakes/trellis to lift foliage off soil.",
        ]
    ),

    # ─────────────────────────────────────────────
    # BACTERIAL DISEASE CAUSES
    # ─────────────────────────────────────────────

    "rainfall_splash_dispersal": CausalHypothesis(
        cause_id="rainfall_splash_dispersal",
        cause_label="Bacterial Dispersal via Rainfall Splash Events",
        disease_families=["BACTERIAL"],
        rainfall_weight=0.70,
        humidity_weight=0.20,
        wetness_weight=0.10,
        rainfall_threshold=15.0,
        humidity_threshold=70.0,
        wetness_threshold=4.0,
        impact=0.55,
        feasibility=0.50,
        action_templates=[
            "Apply copper-based bactericide after rain events (>10mm).",
            "Avoid field work (walking between rows) during wet weather to reduce bacterial spread.",
            "Install wind-breaks or row covers to intercept rain splash.",
        ]
    ),

    "high_temperature_stress_bacterial": CausalHypothesis(
        cause_id="high_temperature_stress_bacterial",
        cause_label="Heat Stress Predisposing Plants to Bacterial Infection",
        disease_families=["BACTERIAL", "ABIOTIC"],
        temperature_weight=0.80,
        humidity_weight=0.20,
        temperature_threshold=35.0,
        humidity_threshold=50.0,
        is_abiotic_consistent=True,
        impact=0.50,
        feasibility=0.55,
        action_templates=[
            "Apply potassium-based foliar spray to improve heat tolerance.",
            "Use shade nets (30% shading) during peak afternoon heat.",
            "Irrigate in the early morning to cool root zone.",
        ]
    ),

    "mechanical_damage_entry_point": CausalHypothesis(
        cause_id="mechanical_damage_entry_point",
        cause_label="Mechanical Wounds Providing Bacterial Entry Points",
        disease_families=["BACTERIAL"],
        humidity_weight=0.30,
        rainfall_weight=0.30,
        humidity_threshold=60.0,
        rainfall_threshold=5.0,
        impact=0.60,
        feasibility=0.75,
        action_templates=[
            "Disinfect pruning tools with 70% alcohol between plants.",
            "Apply wound sealant paste immediately after pruning cuts.",
            "Reduce mechanical harvesting during wet periods.",
        ]
    ),

    # ─────────────────────────────────────────────
    # VIRAL DISEASE CAUSES
    # ─────────────────────────────────────────────

    "insect_vector_proliferation": CausalHypothesis(
        cause_id="insect_vector_proliferation",
        cause_label="Aphid/Whitefly Vector Proliferation in Warm Dry Conditions",
        disease_families=["VIRAL"],
        temperature_weight=0.60,
        humidity_weight=0.20,
        rainfall_weight=0.20,
        temperature_threshold=28.0,
        humidity_threshold=50.0,  # reversed: lower humidity favors vectors
        rainfall_threshold=5.0,
        impact=0.70,
        feasibility=0.65,
        action_templates=[
            "Apply systemic insecticide (imidacloprid) to control aphid/whitefly populations.",
            "Install yellow sticky traps for early vector detection.",
            "Introduce natural predators (ladybugs, lacewings) for biological control.",
            "Remove and destroy heavily infested plant material.",
        ]
    ),

    "drought_stress_susceptibility": CausalHypothesis(
        cause_id="drought_stress_susceptibility",
        cause_label="Drought Stress Reducing Plant Immune Response",
        disease_families=["VIRAL", "ABIOTIC"],
        temperature_weight=0.40,
        rainfall_weight=0.40,
        humidity_weight=0.20,
        temperature_threshold=32.0,
        rainfall_threshold=3.0,  # below threshold = drought indicator
        humidity_threshold=40.0,
        is_abiotic_consistent=True,
        impact=0.60,
        feasibility=0.80,
        action_templates=[
            "Install drip irrigation to maintain consistent soil moisture.",
            "Apply organic mulch (straw, coconut coir) to retain soil moisture.",
            "Apply silicon-based foliar spray to improve drought tolerance.",
        ]
    ),

    # ─────────────────────────────────────────────
    # ABIOTIC STRESS CAUSES
    # ─────────────────────────────────────────────

    "nutrient_deficiency": CausalHypothesis(
        cause_id="nutrient_deficiency",
        cause_label="Nutrient Deficiency Mimicking Disease Symptoms",
        disease_families=["ABIOTIC"],
        humidity_weight=0.10,
        temperature_weight=0.10,
        humidity_threshold=50.0,
        temperature_threshold=20.0,
        is_abiotic_consistent=True,
        impact=0.70,
        feasibility=0.85,
        action_templates=[
            "Conduct soil nutrient analysis to identify specific deficiency (N, P, K, Fe, Mg).",
            "Apply targeted foliar micronutrient spray based on test results.",
            "Incorporate balanced NPK fertilizer at recommended dose.",
        ]
    ),

    "soil_pH_imbalance": CausalHypothesis(
        cause_id="soil_pH_imbalance",
        cause_label="Extreme Soil pH Stress Blocking Nutrient Uptake",
        disease_families=["ABIOTIC"],
        humidity_weight=0.10,
        humidity_threshold=50.0,
        is_abiotic_consistent=True,
        impact=0.65,
        feasibility=0.70,
        action_templates=[
            "Test soil pH — optimal range is 6.0-7.5 for most crops.",
            "Apply agricultural lime (calcium carbonate) to correct acidity.",
            "Apply sulfur or aluminum sulfate to correct alkalinity.",
        ]
    ),
    "water_stress_overwatering": CausalHypothesis(
        cause_id="water_stress_overwatering",
        cause_label="Over-irrigation Creating Anaerobic Root Conditions",
        disease_families=["FUNGAL", "ABIOTIC"],
        rainfall_weight=0.30,
        humidity_weight=0.30,
        wetness_weight=0.40,
        rainfall_threshold=25.0,
        humidity_threshold=75.0,
        wetness_threshold=12.0,
        is_abiotic_consistent=True,
        impact=0.70,
        feasibility=0.85,
        action_templates=[
            "Reduce irrigation frequency — check soil moisture before watering.",
            "Improve soil drainage by raising beds or adding organic matter.",
            "Allow soil to dry between irrigation cycles (stick test: 2 inches dry = time to water).",
        ]
    ),
}


# =============================================================================
# Disease Family Classifier
# =============================================================================

# Keywords mapping disease name fragments to their primary family
DISEASE_FAMILY_KEYWORDS: Dict[str, str] = {
    # BACTERIAL — check before FUNGAL to avoid 'spot' ambiguity
    "bacterial": "BACTERIAL",
    "canker": "BACTERIAL",
    "fire_blight": "BACTERIAL",
    "leaf_burn": "BACTERIAL",
    # VIRAL — check before FUNGAL to avoid 'curl'/'streak' ambiguity
    "virus": "VIRAL",
    "mosaic": "VIRAL",
    "curl": "VIRAL",
    "streak": "VIRAL",
    "yellowing": "VIRAL",
    # FUNGAL
    "blight": "FUNGAL",
    "mold": "FUNGAL",
    "mildew": "FUNGAL",
    "rust": "FUNGAL",
    "scab": "FUNGAL",
    "spot": "FUNGAL",
    "rot": "FUNGAL",
    "anthracnose": "FUNGAL",
    "botrytis": "FUNGAL",
    "powdery": "FUNGAL",
    "downy": "FUNGAL",
    "smut": "FUNGAL",
    "wilt": "FUNGAL",
    # ABIOTIC / HEALTHY
    "deficiency": "ABIOTIC",
    "scorch": "ABIOTIC",
    "healthy": "ABIOTIC",
}


def classify_disease_family(disease_name: str) -> str:
    """
    Classify a disease name string into one of: FUNGAL | BACTERIAL | VIRAL | ABIOTIC.
    Falls back to FUNGAL (most common class in PlantVillage dataset) if no match.
    """
    name_lower = disease_name.lower()
    for keyword, family in DISEASE_FAMILY_KEYWORDS.items():
        if keyword in name_lower:
            return family
    return "FUNGAL"  # safe default


def get_hypotheses_for_disease(disease_name: str) -> Dict[str, CausalHypothesis]:
    """
    Returns all causal hypotheses relevant to the given disease name.
    Uses disease family classification then filters the master ontology.
    """
    family = classify_disease_family(disease_name)
    return {
        cid: hyp
        for cid, hyp in CAUSAL_ONTOLOGY.items()
        if family in hyp.disease_families
    }
