from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any

from backend.models.db_models import User, DiagnosisHistory

async def get_or_create_user(db: AsyncSession, user_id: str, email: str = "unknown@example.com") -> User:
    """
    Fetches a user by ID. If they do not exist (e.g., first time diagnosing after signup),
    creates a shadow record in the local database to satisfy foreign key constraints.
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        user = User(id=user_id, email=email)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
    return user

async def create_diagnosis(
    db: AsyncSession, 
    user_id: str, 
    diagnosis_data: Dict[str, Any]
) -> DiagnosisHistory:
    """
    Persists a new AI diagnosis result into the database.
    """
    db_diagnosis = DiagnosisHistory(
        user_id=user_id,
        disease_name=diagnosis_data.get("disease_name"),
        confidence=diagnosis_data.get("confidence"),
        concern_level=diagnosis_data.get("concern_level"),
        concern_score=diagnosis_data.get("concern_score"),
        temperature=diagnosis_data.get("temperature"),
        humidity=diagnosis_data.get("humidity"),
        ph_level=diagnosis_data.get("ph_level"),
        gradcam_heatmap=diagnosis_data.get("gradcam_heatmap"),
        root_cause_json=diagnosis_data.get("root_cause_json")
    )
    
    db.add(db_diagnosis)
    await db.commit()
    await db.refresh(db_diagnosis)
    
    return db_diagnosis

async def get_user_diagnoses(db: AsyncSession, user_id: str, limit: int = 20) -> List[DiagnosisHistory]:
    """
    Retrieves the diagnosis history for a specific user, ordered by most recent first.
    """
    result = await db.execute(
        select(DiagnosisHistory)
        .where(DiagnosisHistory.user_id == user_id)
        .order_by(DiagnosisHistory.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
