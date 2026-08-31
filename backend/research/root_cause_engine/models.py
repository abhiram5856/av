"""
TRACE-RCE Output Schemas
========================
Pydantic models defining the complete output contract of the Root Cause Engine.
These are standalone schemas that do NOT modify any production domain models.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID, uuid4


class ConfidenceInterval(BaseModel):
    """Represents a calibrated confidence interval for a single root cause."""
    lower: float = Field(..., description="Lower bound of confidence interval (0-1)")
    upper: float = Field(..., description="Upper bound (calibrated point estimate) (0-1)")
    uncertainty_epistemic: float = Field(..., description="Evidence-gap uncertainty (0-1)")
    uncertainty_aleatoric: float = Field(..., description="Measurement noise uncertainty (0-1)")
    uncertainty_total: float = Field(..., description="Combined uncertainty (0-1)")


class EvidenceContribution(BaseModel):
    """Tracks how much each modality contributed to a causal attribution score."""
    modality: str = Field(..., description="Modality name: visual|environmental|severity|knowledge|historical")
    raw_evidence_score: float = Field(..., description="Raw evidence score e_m(c) before weighting (0-1)")
    reliability_weight: float = Field(..., description="Reliability-adjusted weight W_m(c) for this run (0-1)")
    weighted_contribution: float = Field(..., description="W_m(c) × e_m(c) — actual contribution to CAS")


class RootCause(BaseModel):
    """A single ranked root cause with full attribution evidence."""
    rank: int = Field(..., description="Rank (1=highest CAS_final)")
    cause_id: str = Field(..., description="Machine-readable cause identifier (e.g., excessive_leaf_wetness)")
    cause_label: str = Field(..., description="Human-readable cause label for farmer presentation")
    disease_family: str = Field(..., description="Disease family this cause belongs to (FUNGAL|BACTERIAL|VIRAL|ABIOTIC)")
    
    cas_raw: float = Field(..., description="CAS_raw(c) before conflict resolution (0-1)")
    cas_final: float = Field(..., description="CAS_final(c) after conflict resolution (0-1)")
    conflict_discounting_applied: float = Field(default=0.0, description="Amount CAS was discounted by CRL")
    
    confidence: ConfidenceInterval
    evidence_contributions: List[EvidenceContribution] = Field(default_factory=list)
    
    intervention_priority_score: float = Field(..., description="IPS(c) = CAS_final × impact × feasibility (0-1)")
    intervention_impact: float = Field(..., description="Expected severity reduction if cause is addressed (0-1)")
    intervention_feasibility: float = Field(..., description="Ease of farmer action (1=immediate, 0=not actionable)")
    
    reasoning_sentence: str = Field(..., description="Natural language explanation of why this cause scored high")
    actionable_steps: List[str] = Field(default_factory=list, description="Concrete steps farmer can take")


class EvidenceGraphNode(BaseModel):
    """A node in the causal evidence graph."""
    node_id: str
    node_type: str  # 'modality' | 'cause' | 'disease'
    label: str
    score: float = 0.0


class EvidenceGraphEdge(BaseModel):
    """A directed edge in the causal evidence graph."""
    source_id: str
    target_id: str
    weight: float = Field(..., description="Edge weight = weighted contribution or CAS score")
    label: str = ""


class EvidenceGraph(BaseModel):
    """Directed acyclic graph showing evidence flow from modalities to causes to disease."""
    nodes: List[EvidenceGraphNode] = Field(default_factory=list)
    edges: List[EvidenceGraphEdge] = Field(default_factory=list)


class ConflictEvent(BaseModel):
    """Records a detected conflict between two modalities."""
    modality_a: str
    modality_b: str
    cause_id: str
    conflict_score: float
    discounting_applied: float


class RootCauseResult(BaseModel):
    """
    Complete output of the TRACE-RCE Root Cause Engine.
    This is the definitive response contract for NOVA's causal attribution.
    """
    analysis_id: UUID = Field(default_factory=uuid4)
    context_hash: str = Field(..., description="SHA256 hash of the AIContext used as input (for reproducibility)")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    algorithm_version: str = "TRACE-RCE-v1.0"
    
    # Primary outputs
    disease_name: str
    disease_family: str
    ranked_causes: List[RootCause] = Field(default_factory=list)
    
    # Explainability artifacts
    reasoning_chain: List[str] = Field(
        default_factory=list,
        description="Step-by-step causal reasoning narrative for top causes"
    )
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    conflicts_detected: List[ConflictEvent] = Field(default_factory=list)
    
    # Risk and intervention
    top_intervention: Optional[str] = Field(None, description="Single most impactful action for the farmer")
    risk_forecast: str = Field(default="", description="Short-term disease progression risk assessment")
    
    # Metadata
    modality_reliability_scores: Dict[str, float] = Field(default_factory=dict)
    causes_evaluated: int = 0
    causes_above_threshold: int = 0
    analysis_latency_ms: Optional[float] = None
