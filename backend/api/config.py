from fastapi import APIRouter
from typing import Dict, Any
from backend.config.settings import settings

router = APIRouter()

@router.get("/")
async def get_configuration() -> Dict[str, Any]:
    """
    Get current configuration options (excluding secrets).
    """
    config_dict = settings.model_dump(exclude={"DEBUG_MODE"})
    return config_dict
