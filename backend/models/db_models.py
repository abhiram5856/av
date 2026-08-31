import datetime
import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from backend.core.database import Base

class User(Base):
    __tablename__ = "users"

    # We use a string for ID because Supabase provides string UUIDs.
    # Alternatively we can use postgres UUID, but String is safer for generic JWT subs.
    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # A user can have many diagnosis history records
    diagnoses = relationship("DiagnosisHistory", back_populates="user", cascade="all, delete-orphan")

class DiagnosisHistory(Base):
    __tablename__ = "diagnosis_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    
    # Core AI outputs
    disease_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    concern_level = Column(String(50), nullable=True)
    concern_score = Column(Float, nullable=True)
    
    # Environmental Context
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    ph_level = Column(Float, nullable=True)
    
    # Complex/Metadata
    # We store GradCAM heatmaps as Base64 strings or URLs if uploaded to cloud storage
    gradcam_heatmap = Column(String, nullable=True) 
    # The structured JSON output from TRACE-RCE v3
    root_cause_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Establish relationship back to user
    user = relationship("User", back_populates="diagnoses")
