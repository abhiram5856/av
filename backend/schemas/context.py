from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib

@dataclass(frozen=True)
class ImageMetadataContext:
    filename: str
    width: int
    height: int
    format: str
    aspect_ratio: float
    blur_score: float
    is_valid_quality: bool

@dataclass(frozen=True)
class VisionDetectionContext:
    predicted_disease: str
    confidence_score: float
    topk_predictions: Dict[str, float]
    concept_activations: Dict[str, float]

@dataclass(frozen=True)
class GradCAMContext:
    heatmap_coverage_ratio: float
    peak_intensity: float
    target_layer_name: str
    s3_heatmap_url: str

@dataclass(frozen=True)
class ConcernContext:
    base_vision_score: float
    environmental_risk_factor: float
    soil_stress_factor: float
    concern_score: float
    concern_level: str

@dataclass(frozen=True)
class WeatherContext:
    latitude: float
    longitude: float
    temperature_7d_avg: float
    humidity_7d_avg: float
    total_precipitation_mm: float
    leaf_wetness_hours: float
    raw_forecast_summary: Dict[str, Any]

@dataclass(frozen=True)
class RAGKnowledgeContext:
    retrieved_chunk_ids: List[str]
    document_sources: List[str]
    context_text_block: str

@dataclass(frozen=True)
class PatientHistoryContext:
    total_previous_diagnoses: int
    frequent_crop_diseases: List[str]
    last_diagnosis_date: Optional[str] = None

@dataclass(frozen=True)
class UserMetadataContext:
    user_id: str
    role: str
    region: str
    preferred_language: str
    growth_stage: Optional[str] = None

@dataclass(frozen=True)
class SystemMetadataContext:
    schema_version: str = "1.0.0"
    environment: str = "production"
    timestamp_utc: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str = ""

@dataclass(frozen=True)
class AIContext:
    """
    Immutable, sealed multi-modal context object for NOVA.
    Passed directly to downstream analytical interfaces: `engine.analyze(context)`
    """
    system: SystemMetadataContext
    user: UserMetadataContext
    image: ImageMetadataContext
    vision: VisionDetectionContext
    gradcam: GradCAMContext
    concern: ConcernContext
    weather: WeatherContext
    knowledge: RAGKnowledgeContext
    history: PatientHistoryContext
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively converts dataclass to standard Python dictionary."""
        def _convert(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            elif isinstance(obj, list):
                return [_convert(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        return _convert(self)

    def to_json(self) -> str:
        """Serializes immutable context to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def get_content_hash(self) -> str:
        """Computes deterministic SHA256 checksum of context contents."""
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
