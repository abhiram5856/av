from fastapi import APIRouter
from typing import List
from backend.models.domain import ChatMessage
from uuid import UUID

router = APIRouter()

@router.get("/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: UUID, limit: int = 20):
    """
    Retrieve conversation history for a specific session.
    """
    pass
