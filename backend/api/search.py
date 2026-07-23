from fastapi import APIRouter
from typing import List
from backend.models.domain import SearchResult

router = APIRouter()

@router.get("/", response_model=List[SearchResult])
async def search_knowledge_base(query: str, top_k: int = 5):
    """
    Search the vector store directly for a given query.
    """
    pass
