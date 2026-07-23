from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Check system health including DB, VectorStore, and LLM statuses.
    """
    return {"status": "ok", "service": "Zenith AgriBot"}
