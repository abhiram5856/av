from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
import math

from backend.core.database import get_db
from backend.models.db_models import DiagnosisHistory
from backend.auth.security import verify_token
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class IntelligenceObservation(BaseModel):
    id: str
    crop: str
    disease: str
    region: str
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: float
    concernScore: float
    timestamp: datetime
    temperature: Optional[float]
    humidity: Optional[float]
    ph: Optional[float]
    isHealthy: bool

class Alert(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    timestamp: datetime

class IntelligenceDataResponse(BaseModel):
    observations: List[IntelligenceObservation]
    alerts: List[Alert]

def get_region(lat: float, lon: float) -> str:
    # A simple mock reverse geocoder for Indian regions based on lat/lon
    # (Just to categorize for the UI based on coordinates)
    if lat > 28 and lon < 77: return "Punjab"
    if lat > 28 and lon >= 77: return "Haryana"
    if 24 < lat <= 28: return "Uttar Pradesh"
    if 18 < lat <= 24 and lon < 78: return "Maharashtra"
    if 16 < lat <= 19 and lon >= 78: return "Telangana"
    if 13 < lat <= 18 and lon < 78: return "Karnataka"
    if 13 < lat <= 18 and lon >= 78: return "Andhra Pradesh"
    return "Other"

@router.get("/", response_model=IntelligenceDataResponse)
async def get_intelligence_data(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(verify_token)
):
    try:
        # Fetch real records from DB
        result = await db.execute(
            select(DiagnosisHistory)
            .order_by(DiagnosisHistory.created_at.desc())
            .limit(5000)
        )
        records = result.scalars().all()
        
        observations = []
        late_blight_count = 0
        high_concern_count = 0
        
        for r in records:
            # Parse crop from disease name (e.g., tomato_early_blight -> tomato)
            disease_lower = r.disease_name.lower()
            crop = disease_lower.split('_')[0].capitalize()
            is_healthy = "healthy" in disease_lower
            
            # Privacy Protection: Do not expose precise GPS of other users
            lat = r.latitude if r.latitude else None
            lon = r.longitude if r.longitude else None
            
            # If coordinates exist but record belongs to someone else, obfuscate (~1km precision)
            if lat and lon and getattr(r, 'user_id', None) != user_id:
                lat = round(lat, 2)
                lon = round(lon, 2)
                
            region = get_region(lat, lon) if (lat and lon) else "Unknown"
            
            obs = IntelligenceObservation(
                id=r.id,
                crop=crop,
                disease=r.disease_name,
                region=region,
                latitude=lat,
                longitude=lon,
                confidence=r.confidence,
                concernScore=r.concern_score or 50.0,
                timestamp=r.created_at,
                temperature=r.temperature,
                humidity=r.humidity,
                ph=r.ph_level,
                isHealthy=is_healthy
            )
            observations.append(obs)
            
            if "late_blight" in disease_lower:
                late_blight_count += 1
            if r.concern_score and r.concern_score >= 80:
                high_concern_count += 1

        alerts = []
        if late_blight_count > 5:
            alerts.append(Alert(
                id="alert-late-blight",
                title="Late Blight Outbreak Detected",
                description=f"Unusually high incidence ({late_blight_count} cases) of Late Blight detected recently. Agronomists should issue preemptive fungicide warnings.",
                severity="critical",
                timestamp=datetime.utcnow()
            ))
            
        if high_concern_count > 10:
            alerts.append(Alert(
                id="alert-high-concern",
                title="Elevated Disease Severity",
                description=f"{high_concern_count} high-concern diagnoses recorded. Immediate field reviews recommended.",
                severity="warning",
                timestamp=datetime.utcnow()
            ))

        return IntelligenceDataResponse(
            observations=observations,
            alerts=alerts
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to aggregate intelligence data")
