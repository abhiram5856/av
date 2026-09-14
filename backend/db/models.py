import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # A user can have multiple diagnosis records
    diagnoses = relationship("DiagnosisRecord", back_populates="user")

class DiagnosisRecord(Base):
    __tablename__ = 'diagnosis_records'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), index=True)
    
    # ML details
    disease_name = Column(String, index=True)
    confidence = Column(Float)
    severity_assessment = Column(JSON) # Store humidity, temp, ph, lesion area, etc.
    
    # Store the s3 bucket URL or path to the uploaded image / gradcam heatmap
    image_url = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Backref to user
    user = relationship("User", back_populates="diagnoses")
