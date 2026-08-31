from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from backend.auth.security import verify_token
from backend.api.iot import get_latest_reading

router = APIRouter()

class EnvironmentalRiskEstimate(BaseModel):
    fungal_risk: str
    fungal_risk_reason: str
    irrigation_status: str
    irrigation_reason: str
    overall_health: str
    is_heuristic: bool = True

@router.get("/", response_model=EnvironmentalRiskEstimate)
async def estimate_environmental_risk(user_id: str = Depends(verify_token)):
    """
    Lightweight heuristic risk estimation based on recent IoT context.
    Provides practical precision-farming insights without faking a complex AI model.
    """
    latest_reading = await get_latest_reading(user_id)
    
    # Fungal Risk Logic
    if latest_reading.humidity > 75.0 and latest_reading.soil_moisture > 60.0:
        fungal_risk = "Moderate → Increasing"
        fungal_reason = "Humidity and soil moisture have remained high."
    else:
        fungal_risk = "Low"
        fungal_reason = "Environmental conditions are not currently supportive for fungal growth."
        
    # Irrigation Logic
    if latest_reading.soil_moisture < 40.0:
        irrigation = "Requires Attention"
        irrigation_reason = "Soil moisture is trending downward."
    elif latest_reading.soil_moisture > 80.0:
        irrigation = "Excessive"
        irrigation_reason = "Soil moisture is very high. Avoid watering."
    else:
        irrigation = "Adequate"
        irrigation_reason = "Soil moisture is within the optimal range."
        
    # Overall
    if fungal_risk == "Low" and irrigation == "Adequate":
        overall = "Currently Stable"
    else:
        overall = "Monitor Closely"
        
    return EnvironmentalRiskEstimate(
        fungal_risk=fungal_risk,
        fungal_risk_reason=fungal_reason,
        irrigation_status=irrigation,
        irrigation_reason=irrigation_reason,
        overall_health=overall,
        is_heuristic=True
    )
