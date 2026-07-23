from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    text: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chunk_id: UUID
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DiseasePrediction(BaseModel):
    disease_name: str
    confidence: float
    description: Optional[str] = None
    treatment_recommendations: Optional[List[str]] = None

class ChatMessage(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: str = Field(..., description="user or assistant")
    content: str
    language: str = Field(default="en")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserQuery(BaseModel):
    query: str
    language: Optional[str] = None
    conversation_id: Optional[UUID] = None
    disease_context: Optional[DiseasePrediction] = None

class ChatResponse(BaseModel):
    response: str
    language: str
    source_chunks: List[SearchResult] = Field(default_factory=list)
    processing_time_ms: Optional[int] = None
