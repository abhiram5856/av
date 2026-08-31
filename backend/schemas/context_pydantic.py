from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

class ImageMetadataDTO(StrictBaseModel):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str
    aspect_ratio: float
    blur_score: float = Field(ge=0.0)
    is_valid_quality: bool

class VisionDetectionDTO(StrictBaseModel):
    predicted_disease: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    topk_predictions: Dict[str, float]
    concept_activations: Dict[str, float]

class GradCAMDTO(StrictBaseModel):
    heatmap_coverage_ratio: float = Field(ge=0.0, le=1.0)
    peak_intensity: float = Field(ge=0.0, le=1.0)
    target_layer_name: str
    s3_heatmap_url: str

class ConcernDTO(StrictBaseModel):
    base_vision_score: float = Field(ge=0.0, le=100.0)
    environmental_risk_factor: float = Field(ge=1.0, le=2.0)
    soil_stress_factor: float = Field(ge=1.0, le=2.0)
    concern_score: float = Field(ge=0.0, le=100.0)
    concern_level: str

class WeatherDTO(StrictBaseModel):
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    temperature_7d_avg: float
    humidity_7d_avg: float = Field(ge=0.0, le=100.0)
    total_precipitation_mm: float = Field(ge=0.0)
    leaf_wetness_hours: float = Field(ge=0.0, le=24.0)
    raw_forecast_summary: Dict[str, Any]

class RAGKnowledgeDTO(StrictBaseModel):
    retrieved_chunk_ids: List[str]
    document_sources: List[str]
    context_text_block: str

class PatientHistoryDTO(StrictBaseModel):
    total_previous_diagnoses: int = Field(ge=0)
    frequent_crop_diseases: List[str]
    last_diagnosis_date: Optional[str] = None

class UserMetadataDTO(StrictBaseModel):
    user_id: str
    role: str
    region: str
    preferred_language: str

class SystemMetadataDTO(StrictBaseModel):
    schema_version: str = "1.0.0"
    environment: str = "production"
    timestamp_utc: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str

class AIContextPydantic(StrictBaseModel):
    system: SystemMetadataDTO
    user: UserMetadataDTO
    image: ImageMetadataDTO
    vision: VisionDetectionDTO
    gradcam: GradCAMDTO
    concern: ConcernDTO
    weather: WeatherDTO
    knowledge: RAGKnowledgeDTO
    history: PatientHistoryDTO
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
