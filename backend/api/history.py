from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime

from backend.core.database import get_db
from backend.data.crud import get_user_diagnoses
from backend.auth.security import verify_token

router = APIRouter()

# Pydantic schema for serializing output
class DiagnosisResponse(BaseModel):
    id: str
    disease_name: str
    confidence: float
    concern_level: str | None
    concern_score: float | None
    temperature: float | None
    humidity: float | None
    ph_level: float | None
    gradcam_heatmap: str | None
    root_cause_json: Dict[str, Any] | None
    created_at: datetime

    class Config:
        from_attributes = True

@router.get("/", response_model=List[DiagnosisResponse])
async def get_user_history(limit: int = 20, db: AsyncSession = Depends(get_db), user_id: str = Depends(verify_token)):
    """
    Retrieve diagnosis history for the authenticated user from the PostgreSQL database.
    """
    try:
        diagnoses = await get_user_diagnoses(db, user_id=user_id, limit=limit)
        return diagnoses
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve history")
