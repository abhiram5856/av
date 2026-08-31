"""
TRACE-RCE Evaluation Dataset Generator
======================================
Phase 3.1 — Dataset Expansion

Programmatically generates a comprehensive synthetic evaluation dataset of exactly 1050
realistic case scenarios (AIContext instances) covering:
  - 15 disease/crop combinations
  - Fungal, Bacterial, Viral, and Abiotic families
  - Different weather conditions (conducive, dry, extreme)
  - Modality deficiencies (missing weather, blurry images, missing RAG)
  - Conflicting evidence profiles

Each case has:
  - A unique case_id (e.g., CASE-001)
  - Fully populated immutable AIContext
  - Ground truth primary and secondary root cause IDs
  - Metadata tags (crop, family, weather, quality, history, perturbation)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from backend.schemas.context import (
    AIContext, SystemMetadataContext, UserMetadataContext, ImageMetadataContext,
    VisionDetectionContext, GradCAMContext, ConcernContext, WeatherContext,
    RAGKnowledgeContext, PatientHistoryContext
)


@dataclass
class EvalScenario:
    """Represents a generated evaluation scenario with ground truth."""
    case_id: str
    description: str
    context: AIContext
    ground_truth_causes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Helper function to construct AIContext cleanly
# =============================================================================

def _build_context(
    request_id: str,
    disease: str,
    confidence: float,
    coverage: float,
    peak: float,
    humidity: float,
    temp: float,
    rainfall: float,
    wetness: float,
    env_factor: float,
    soil_factor: float,
    severity: float,
    rag_chunks: List[str],
    rag_text: str,
    total_hist: int,
    frequent: List[str],
    last_date: Optional[str] = None,
    blur_score: float = 85.0,
    is_valid: bool = True,
) -> AIContext:
    """Fluent constructor for evaluation context."""
    return AIContext(
        system=SystemMetadataContext(
            request_id=request_id,
            environment="evaluation"
        ),
        user=UserMetadataContext(
            user_id="eval-user",
            role="farmer",
            region="evaluation-field",
            preferred_language="en"
        ),
        image=ImageMetadataContext(
            filename=f"{request_id}_leaf.jpg",
            width=224,
            height=224,
            format="JPEG",
            aspect_ratio=1.0,
            blur_score=blur_score,
            is_valid_quality=is_valid
        ),
        vision=VisionDetectionContext(
            predicted_disease=disease,
            confidence_score=confidence,
            topk_predictions={disease: confidence, "healthy": 1.0 - confidence},
            concept_activations={}
        ),
        gradcam=GradCAMContext(
            heatmap_coverage_ratio=coverage,
            peak_intensity=peak,
            target_layer_name="features.13",
            s3_heatmap_url=f"http://eval-s3/{request_id}_heatmap.jpg"
        ),
        severity=ConcernContext(
            base_vision_score=round(severity * 0.6, 2),
            environmental_risk_factor=env_factor,
            soil_stress_factor=soil_factor,
            concern_score=severity,
            concern_level="High" if severity >= 70.0 else "Medium" if severity >= 30.0 else "Low"
        ),
        weather=WeatherContext(
            latitude=17.385,
            longitude=78.486,
            temperature_7d_avg=temp,
            humidity_7d_avg=humidity,
            total_precipitation_mm=rainfall,
            leaf_wetness_hours=wetness,
            raw_forecast_summary={"description": "Evaluated Weather Data"}
        ),
        knowledge=RAGKnowledgeContext(
            retrieved_chunk_ids=rag_chunks,
            document_sources=["eval_pathology_manual.txt"],
            context_text_block=rag_text
        ),
        history=PatientHistoryContext(
            total_previous_diagnoses=total_hist,
            frequent_crop_diseases=frequent,
            last_diagnosis_date=last_date
        )
    )


# =============================================================================
# Crop-Disease Configuration Map (Ground truths & standard RAG text)
# =============================================================================

CROP_DISEASES = [
    # FUNGAL
    {"disease": "tomato_late_blight", "crop": "tomato", "family": "FUNGAL", 
     "primary": "excessive_leaf_wetness", "secondary": "high_humidity_conduciveness",
     "rag": "Late blight of tomato is caused by the oomycete Phytophthora infestans. It thrives under cool, wet conditions, particularly when leaf wetness duration exceeds 8 to 12 hours. High relative humidity above 80% is critical for spore generation and infection."},
    
    {"disease": "tomato_early_blight", "crop": "tomato", "family": "FUNGAL", 
     "primary": "high_humidity_conduciveness", "secondary": "poor_air_circulation",
     "rag": "Early blight is caused by Alternaria solani. It manifests under alternating wet and dry conditions in the canopy. Poor air circulation and crowded plant spacing traps moisture, intensifying early blight lesion spread."},
    
    {"disease": "corn_northern_leaf_blight", "crop": "corn", "family": "FUNGAL", 
     "primary": "temperature_optimal_for_fungus", "secondary": "poor_air_circulation",
     "rag": "Northern corn leaf blight (Exserohilum turcicum) is favored by moderate temperatures between 20C and 28C, heavy dews, and poor air circulation within dense canopies."},
    
    {"disease": "wheat_rust", "crop": "wheat", "family": "FUNGAL", 
     "primary": "high_humidity_conduciveness", "secondary": "temperature_optimal_for_fungus",
     "rag": "Wheat rust species require free water or high relative humidity to germinate. Optimal temperatures for rust development are 18C to 25C, where spore germination and leaf penetration occur rapidly."},
    
    {"disease": "potato_late_blight", "crop": "potato", "family": "FUNGAL", 
     "primary": "excessive_leaf_wetness", "secondary": "soil_borne_pathogen_carryover",
     "rag": "Late blight in potato is driven by excessive leaf wetness duration. Soil-borne pathogen carryover in volunteer tubers or infected crop debris from previous seasons is the major source of primary inoculum."},
    
    {"disease": "grape_black_rot", "crop": "grape", "family": "FUNGAL", 
     "primary": "excessive_leaf_wetness", "secondary": "high_humidity_conduciveness",
     "rag": "Black rot of grapes (Guignardia bidwellii) requires wet leaves for infection to occur. Warm temperatures and long wetness periods lead to severe black rot development."},

    # BACTERIAL
    {"disease": "tomato_bacterial_spot", "crop": "tomato", "family": "BACTERIAL", 
     "primary": "rainfall_splash_dispersal", "secondary": "mechanical_damage_entry_point",
     "rag": "Bacterial spot is caused by Xanthomonas species. Bacteria are dispersed between plants primarily by wind-driven rain splash events. Mechanical damage and pruning wounds provide main entry points."},
    
    {"disease": "pepper_bacterial_spot", "crop": "pepper", "family": "BACTERIAL", 
     "primary": "rainfall_splash_dispersal", "secondary": "mechanical_damage_entry_point",
     "rag": "Bacterial spot on pepper spreads quickly in heavy rain splash events. Over-irrigation can lead to severe bacterial leaf spot by increasing splash transport of bacteria."},

    {"disease": "apple_fire_blight", "crop": "apple", "family": "BACTERIAL", 
     "primary": "mechanical_damage_entry_point", "secondary": "high_temperature_stress_bacterial",
     "rag": "Fire blight (Erwinia amylovora) attacks apple blossoms and shoots. Wounds from pruning tools or insect entry are critical gateways. High temperature stress makes plants susceptible."},

    # VIRAL
    {"disease": "tomato_yellow_leaf_curl_virus", "crop": "tomato", "family": "VIRAL", 
     "primary": "insect_vector_proliferation", "secondary": "drought_stress_susceptibility",
     "rag": "TYLCV is a begomovirus transmitted exclusively by the whitefly Bemisia tabaci. Whiteflies proliferate rapidly in warm, dry weather conditions. Drought stress lowers plant defenses and accelerates viral spread."},
    
    {"disease": "squash_mosaic_virus", "crop": "squash", "family": "VIRAL", 
     "primary": "insect_vector_proliferation", "secondary": "drought_stress_susceptibility",
     "rag": "Squash mosaic virus is spread by beetle vectors. Heavy vector populations occur in hot, dry climates. Water stress increases leaf temperature, attracting more vectors."},

    # ABIOTIC
    {"disease": "tomato_nutrient_deficiency", "crop": "tomato", "family": "ABIOTIC", 
     "primary": "nutrient_deficiency", "secondary": "soil_pH_imbalance",
     "rag": "Yellowing and necrosis on tomato leaves can represent nitrogen or magnesium deficiency. Extreme soil pH levels impede nutrient uptake by roots even if fertilizer is present."},
    
    {"disease": "corn_drought_stress", "crop": "corn", "family": "ABIOTIC", 
     "primary": "drought_stress_susceptibility", "secondary": "nutrient_deficiency",
     "rag": "Drought stress in corn results from prolonged dry conditions and lack of rainfall. Water deficit limits transpiration and nutrient transport, mimicking visual disease symptoms."},

    {"disease": "bean_soil_acidity_stress", "crop": "bean", "family": "ABIOTIC", 
     "primary": "soil_pH_imbalance", "secondary": "nutrient_deficiency",
     "rag": "Acid soils with low pH limit bean root development. It blocks phosphorus and calcium uptake, causing interveinal chlorosis and stunted growth."}
]


# =============================================================================
# Dataset Generator Implementation
# =============================================================================

def generate_evaluation_dataset(num_scenarios: int = 1050) -> List[EvalScenario]:
    """
    Generates structured synthetic evaluation cases.
    We iterate over combinations of crop diseases, environmental conditions,
    image quality levels, and data perturbations to fill the target size.
    
    WARNING (SCIENTIFIC RIGOR): 
    This dataset is PROGRAMMATICALLY GENERATED. Ground-truth causes are assigned
    by deterministic heuristic rules, not empirical epidemiological studies.
    Models trained on this dataset will learn these specific human-authored heuristics.
    """
    scenarios: List[EvalScenario] = []
    case_counter = 1

    # Define standard combinations
    weather_scenarios = [
        # (humidity, temp, rainfall, wetness)
        {"name": "conducive_fungal", "hum": 88.0, "temp": 22.0, "rain": 20.0, "wet": 12.0},
        {"name": "conducive_bacterial", "hum": 75.0, "temp": 30.0, "rain": 25.0, "wet": 6.0},
        {"name": "dry_vector", "hum": 35.0, "temp": 33.0, "rain": 2.0, "wet": 1.0},
        {"name": "neutral", "hum": 60.0, "temp": 24.0, "rain": 5.0, "wet": 4.0},
        {"name": "cold_wet", "hum": 90.0, "temp": 12.0, "rain": 30.0, "wet": 16.0},
    ]

    quality_scenarios = [
        {"name": "excellent", "blur": 95.0, "valid": True},
        {"name": "degraded", "blur": 45.0, "valid": True},
        {"name": "blurry_invalid", "blur": 8.0, "valid": False},
    ]

    history_scenarios = [
        {"name": "none", "total": 0, "frequent": []},
        {"name": "moderate", "total": 5, "frequent": ["tomato_early_blight"]},
        {"name": "rich", "total": 15, "frequent": ["tomato_late_blight", "early_blight"]},
    ]

    # We systematically loop to generate the requested number of cases.
    for i in range(num_scenarios):
        config_idx = i % len(CROP_DISEASES)
        cfg = CROP_DISEASES[config_idx]

        w_idx = (i // len(CROP_DISEASES)) % len(weather_scenarios)
        w_cfg = weather_scenarios[w_idx]

        q_idx = (i // (len(CROP_DISEASES) * len(weather_scenarios))) % len(quality_scenarios)
        q_cfg = quality_scenarios[q_idx]

        h_idx = (i // (len(CROP_DISEASES) * len(weather_scenarios) * len(quality_scenarios))) % len(history_scenarios)
        h_cfg = history_scenarios[h_idx]

        perturbation = "none"
        disease = cfg["disease"]
        confidence = 0.85 + (i % 15) * 0.01  # range 0.85 to 0.99
        coverage = 0.25 + (i % 10) * 0.05    # range 0.25 to 0.70
        peak = 0.60 + (i % 8) * 0.05         # range 0.60 to 0.95
        
        humidity = w_cfg["hum"]
        temp = w_cfg["temp"]
        rainfall = w_cfg["rain"]
        wetness = w_cfg["wet"]
        
        env_factor = 1.25 if cfg["family"] == "FUNGAL" else 1.05
        soil_factor = 1.15 if cfg["family"] == "ABIOTIC" else 1.0
        severity = 40.0 + (i % 11) * 4.0      # range 40 to 80
        
        rag_chunks = [f"chk_{i}_1", f"chk_{i}_2"]
        rag_text = cfg["rag"]
        
        # Adjust based on perturbation types (distribute evenly)
        p_type = i % 10
        if p_type == 5:
            perturbation = "missing_weather"
            humidity = 0.0
            temp = 0.0
            rainfall = 0.0
            wetness = 0.0
        elif p_type == 6:
            perturbation = "missing_rag"
            rag_chunks = []
            rag_text = ""
        elif p_type == 7:
            perturbation = "contradictory_evidence"
            # Force environmental variables opposite to what the disease family likes
            if cfg["family"] in ("FUNGAL", "BACTERIAL"):
                humidity = 20.0
                temp = 10.0
                rainfall = 0.0
                wetness = 0.0
                env_factor = 0.7
            else:  # VIRAL or ABIOTIC
                humidity = 90.0
                temp = 15.0
                rainfall = 40.0
                wetness = 16.0
        elif p_type == 8:
            perturbation = "low_confidence"
            confidence = 0.35 + (i % 10) * 0.01  # range 0.35 to 0.44
        elif p_type == 9:
            perturbation = "ood_disease"
            disease = f"unknown_{cfg['crop']}_leaf_anomaly"
            rag_text = f"An unclassified disease spot detected on {cfg['crop']} leaves."

        # Assign ground truth based on crop disease configuration
        g_truth = [cfg["primary"], cfg["secondary"]]

        # Build case ID and description
        case_id = f"CASE-{case_counter:03d}"
        description = (
            f"{cfg['crop'].title()} {cfg['disease'].replace('_', ' ')}: "
            f"weather={w_cfg['name']}, quality={q_cfg['name']}, "
            f"history={h_cfg['name']}, perturbation={perturbation}"
        )

        ctx = _build_context(
            request_id=f"req-{case_counter:03d}",
            disease=disease,
            confidence=confidence,
            coverage=coverage,
            peak=peak,
            humidity=humidity,
            temp=temp,
            rainfall=rainfall,
            wetness=wetness,
            env_factor=env_factor,
            soil_factor=soil_factor,
            severity=severity,
            rag_chunks=rag_chunks,
            rag_text=rag_text,
            total_hist=h_cfg["total"],
            frequent=h_cfg["frequent"],
            last_date="2026-07-20T12:00:00" if h_cfg["total"] > 0 else None,
            blur_score=q_cfg["blur"],
            is_valid=q_cfg["valid"]
        )

        metadata = {
            "case_id": case_id,
            "crop": cfg["crop"],
            "disease_family": cfg["family"],
            "weather_scenario": w_cfg["name"],
            "image_quality": q_cfg["name"],
            "history_depth": h_cfg["name"],
            "perturbation": perturbation,
        }

        scenarios.append(EvalScenario(
            case_id=case_id,
            description=description,
            context=ctx,
            ground_truth_causes=g_truth,
            metadata=metadata
        ))
        
        case_counter += 1

    return scenarios
