from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict

class SourceReference(BaseModel):
    organization: str
    url: Optional[HttpUrl] = None
    date_accessed: Optional[str] = None
    publication_date: Optional[str] = None
    confidence_in_source: str = "High"

class EnvironmentalContext(BaseModel):
    temperature_context: Optional[str] = None
    humidity_context: Optional[str] = None
    rainfall_wetness_context: Optional[str] = None
    soil_context: Optional[str] = None
    growth_stage_context: Optional[str] = None

class DiseaseKnowledge(BaseModel):
    canonical_class: str
    crop: str
    disease_name: str
    common_names: List[str] = []
    scientific_name: Optional[str] = None
    
    disease_type: str = Field(..., description="fungal, bacterial, viral, insect/pest-associated, physiological/disorder, other")
    pathogen_or_cause: Optional[str] = None
    
    primary_plant_parts_affected: List[str] = []
    leaf_symptoms: Optional[str] = None
    stem_symptoms: Optional[str] = None
    fruit_pod_tuber_symptoms: Optional[str] = None
    
    early_symptoms: Optional[str] = None
    advanced_symptoms: Optional[str] = None
    visual_differentiation: Optional[str] = None
    lookalike_conditions: List[str] = []
    
    disease_cycle_summary: Optional[str] = None
    spread_mechanism: Optional[str] = None
    favorable_environmental_conditions: Optional[EnvironmentalContext] = None
    
    monitoring_guidance: Optional[str] = None
    prevention: Optional[str] = None
    cultural_management: Optional[str] = None
    sanitation: Optional[str] = None
    water_irrigation_management: Optional[str] = None
    crop_rotation: Optional[str] = None
    resistant_varieties: Optional[str] = None
    biological_control: Optional[str] = None
    chemical_control: Optional[str] = None
    
    expert_escalation_conditions: Optional[str] = None
    confidence_limitations: Optional[str] = None
    
    source_references: List[SourceReference] = []

class KnowledgeBase(BaseModel):
    diseases: Dict[str, DiseaseKnowledge]
