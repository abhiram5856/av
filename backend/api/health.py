from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "healthy", "service": "NOVA Vision Platform"}

@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    return {"status": "ready"}

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    return {"status": "live"}

@router.get("/version")
async def get_version() -> Dict[str, Any]:
    return {
        "application_version": "1.0.0",
        "model_version": "1.0.0",
        "dataset_version": "PlantVillage-Preprocessed-v1.0",
        "schema_version": "1.0.0"
    }
