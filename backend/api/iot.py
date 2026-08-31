from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import random
import uuid
from backend.auth.security import verify_token

router = APIRouter()

class SensorReadingBase(BaseModel):
    device_id: str
    soil_moisture: float
    temperature: float
    humidity: float
    soil_ph: float
    light_intensity: float

class SensorReading(SensorReadingBase):
    id: str
    user_id: str
    timestamp: datetime
    is_simulated: bool = True

# In-memory store for prototype
_iot_store: List[SensorReading] = []

@router.post("/readings", response_model=SensorReading)
async def submit_reading(reading: SensorReadingBase, user_id: str = Depends(verify_token)):
    """Submit a new sensor reading."""
    new_reading = SensorReading(
        id=str(uuid.uuid4()),
        user_id=user_id,
        timestamp=datetime.utcnow(),
        is_simulated=True,
        **reading.model_dump()
    )
    _iot_store.append(new_reading)
    return new_reading

@router.get("/readings", response_model=List[SensorReading])
async def get_readings(user_id: str = Depends(verify_token), limit: int = 24):
    """Get history of sensor readings for the user."""
    # Enforce isolation: filter by user_id
    user_readings = [r for r in _iot_store if r.user_id == user_id]
    user_readings.sort(key=lambda x: x.timestamp, reverse=True)
    return user_readings[:limit]

@router.get("/latest", response_model=SensorReading)
async def get_latest_reading(user_id: str = Depends(verify_token)):
    """Get the most recent sensor reading for the user."""
    user_readings = [r for r in _iot_store if r.user_id == user_id]
    if not user_readings:
        # For the MVP, if no reading exists, return a simulated one
        return SensorReading(
            id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.utcnow(),
            device_id="sim-device-01",
            soil_moisture=random.uniform(30.0, 70.0),
            temperature=random.uniform(20.0, 35.0),
            humidity=random.uniform(40.0, 85.0),
            soil_ph=random.uniform(5.5, 7.5),
            light_intensity=random.uniform(400.0, 800.0),
            is_simulated=True
        )
    user_readings.sort(key=lambda x: x.timestamp, reverse=True)
    return user_readings[0]
