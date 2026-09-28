from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class IoTSensorReading(BaseModel):
    sensor_id: str
    farm_id: str
    timestamp: datetime
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    variable: str = Field(..., description="air_temperature, relative_humidity, soil_moisture, soil_temperature, soil_ph, rain_gauge, leaf_wetness, light_intensity, wind_speed, ec")
    value: float
    unit: str
    quality_status: str = Field("GOOD", description="GOOD, BAD, SUSPECT, CALIBRATING")
    source_type: str = Field("LIVE_SENSOR", description="LIVE_SENSOR, USER_INPUT, SIMULATED, IMPORTED")
