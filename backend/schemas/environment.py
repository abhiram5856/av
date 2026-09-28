from pydantic import BaseModel, Field
from typing import Optional, List

class WeatherObservation(BaseModel):
    temperature: Optional[float] = None
    relative_humidity: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed: Optional[float] = None
    is_live: bool = False
    source: str = "UNAVAILABLE"

class EnvironmentalCompatibility(BaseModel):
    status: str = Field("INSUFFICIENT DATA", description="SUPPORTIVE, CONFLICTING, INSUFFICIENT DATA")
    confidence: str = Field("Low", description="Low, Moderate, High")
    explanation: str = ""

class LocationIntelligence(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region_name: Optional[str] = None
    source: str = "UNAVAILABLE" # GPS, USER_SELECTED, FALLBACK

class SoilIntelligence(BaseModel):
    ph_level: Optional[float] = None
    source: str = "UNAVAILABLE" # USER_PROVIDED, SENSOR, EXTERNAL, DEFAULT
