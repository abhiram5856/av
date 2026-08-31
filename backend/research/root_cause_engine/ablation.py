"""
TRACE-RCE Ablation Study Module
===============================
Phase 3.4 — Ablation

Provides utilities to systematically ablate single evidence modalities
from an AIContext.
"""

from __future__ import annotations
from backend.schemas.context import (
    AIContext, GradCAMContext, WeatherContext, ConcernContext,
    RAGKnowledgeContext, PatientHistoryContext
)

ABLATION_MODES = [
    "full",
    "ablate_visual",
    "ablate_environmental",
    "ablate_severity",
    "ablate_knowledge",
    "ablate_historical",
]


def ablate_context(context: AIContext, mode: str) -> AIContext:
    """
    Returns a modified copy of the AIContext where the specified modality
    information is completely nullified or set to neutral defaults.
    """
    if mode == "full":
        return context

    # Copy sections
    gradcam = context.gradcam
    weather = context.weather
    severity = context.concern
    knowledge = context.knowledge
    history = context.history

    if mode == "ablate_visual":
        # Remove GradCAM spatial coverage and peak intensity
        gradcam = GradCAMContext(
            heatmap_coverage_ratio=0.0,
            peak_intensity=0.0,
            target_layer_name=context.gradcam.target_layer_name,
            s3_heatmap_url=""
        )
    elif mode == "ablate_environmental":
        # Neutralize environmental weather signals (set to non-conducive defaults)
        weather = WeatherContext(
            latitude=context.weather.latitude,
            longitude=context.weather.longitude,
            temperature_7d_avg=25.0,  # neutral
            humidity_7d_avg=50.0,     # neutral/dry
            total_precipitation_mm=0.0,
            leaf_wetness_hours=0.0,
            raw_forecast_summary={}
        )
    elif mode == "ablate_severity":
        # Zero out severity scores and multipliers
        severity = ConcernContext(
            base_vision_score=0.0,
            environmental_risk_factor=1.0,
            soil_stress_factor=1.0,
            concern_score=0.0,
            concern_level="Low"
        )
    elif mode == "ablate_knowledge":
        # Empty the retrieved knowledge block
        knowledge = RAGKnowledgeContext(
            retrieved_chunk_ids=[],
            document_sources=[],
            context_text_block=""
        )
    elif mode == "ablate_historical":
        # Zero out patient history
        history = PatientHistoryContext(
            total_previous_diagnoses=0,
            frequent_crop_diseases=[]
        )

    return AIContext(
        system=context.system,
        user=context.user,
        image=context.image,
        vision=context.vision,
        gradcam=gradcam,
        severity=severity,
        weather=weather,
        knowledge=knowledge,
        history=history,
        custom_metadata=context.custom_metadata
    )
