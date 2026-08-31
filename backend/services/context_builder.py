from backend.schemas.context import (
    AIContext, SystemMetadataContext, UserMetadataContext, ImageMetadataContext,
    VisionDetectionContext, GradCAMContext, ConcernContext, WeatherContext,
    RAGKnowledgeContext, PatientHistoryContext
)
from typing import Dict, Any, List

class AIContextBuilder:
    """
    Fluent Builder for constructing immutable AIContext instances safely.
    Ensures mandatory blocks are populated before finalizing build.
    """
    def __init__(self, request_id: str, user_id: str):
        self._request_id = request_id
        self._user_id = user_id
        
        self._system = SystemMetadataContext(request_id=request_id)
        self._user = UserMetadataContext(user_id=user_id, role="farmer", region="default", preferred_language="en")
        self._image = None
        self._vision = None
        self._gradcam = None
        self._concern = None
        self._weather = None
        self._knowledge = None
        self._history = PatientHistoryContext(total_previous_diagnoses=0, frequent_crop_diseases=[])
        self._custom: Dict[str, Any] = {}

    def set_user_info(self, role: str, region: str, language: str, growth_stage: str = None):
        self._user = UserMetadataContext(
            user_id=self._user_id, role=role, region=region, preferred_language=language, growth_stage=growth_stage
        )
        return self

    def set_image_metadata(self, filename: str, w: int, h: int, fmt: str, blur_score: float, is_valid: bool):
        ratio = round(w / h, 2) if h > 0 else 1.0
        self._image = ImageMetadataContext(
            filename=filename, width=w, height=h, format=fmt,
            aspect_ratio=ratio, blur_score=blur_score, is_valid_quality=is_valid
        )
        return self

    def set_vision_results(self, disease: str, confidence: float, topk: Dict[str, float], concepts: Dict[str, float]):
        self._vision = VisionDetectionContext(
            predicted_disease=disease, confidence_score=confidence,
            topk_predictions=topk, concept_activations=concepts
        )
        return self

    def set_gradcam_results(self, coverage: float, peak: float, layer: str, url: str):
        self._gradcam = GradCAMContext(
            heatmap_coverage_ratio=coverage, peak_intensity=peak,
            target_layer_name=layer, s3_heatmap_url=url
        )
        return self

    def set_concern_results(self, base: float, env_factor: float, soil_factor: float, concern_score: float, concern_level: str):
        self._concern = ConcernContext(
            base_vision_score=base, environmental_risk_factor=env_factor,
            soil_stress_factor=soil_factor, concern_score=concern_score,
            concern_level=concern_level
        )
        return self

    def set_weather_data(self, lat: float, lon: float, temp_avg: float, hum_avg: float, precip: float, wetness: float, raw: Dict[str, Any]):
        self._weather = WeatherContext(
            latitude=lat, longitude=lon, temperature_7d_avg=temp_avg,
            humidity_7d_avg=hum_avg, total_precipitation_mm=precip,
            leaf_wetness_hours=wetness, raw_forecast_summary=raw
        )
        return self

    def set_knowledge_context(self, chunk_ids: List[str], sources: List[str], text_block: str):
        self._knowledge = RAGKnowledgeContext(
            retrieved_chunk_ids=chunk_ids, document_sources=sources, context_text_block=text_block
        )
        return self

    def set_patient_history(self, total_count: int, frequent: List[str], last_date: str = None):
        self._history = PatientHistoryContext(
            total_previous_diagnoses=total_count, frequent_crop_diseases=frequent, last_diagnosis_date=last_date
        )
        return self

    def build(self) -> AIContext:
        """Validates presence and produces sealed AIContext."""
        if not all([self._image, self._vision, self._gradcam, self._concern, self._weather, self._knowledge]):
            raise ValueError("Cannot build AIContext: One or more required multi-modal blocks are uninitialized.")

        return AIContext(
            system=self._system,
            user=self._user,
            image=self._image,
            vision=self._vision,
            gradcam=self._gradcam,
            concern=self._concern,
            weather=self._weather,
            knowledge=self._knowledge,
            history=self._history,
            custom_metadata=self._custom
        )
