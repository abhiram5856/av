from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel
from backend.schemas.context import AIContext
from backend.research.root_cause_engine.models import RootCauseResult

class AbstractRootCauseEngine(ABC):
    """
    Abstract Interface Contract for NOVA Root Cause Engine.
    DO NOT IMPLEMENT RESEARCH ALGORITHM HERE.
    Receives ONLY the immutable AIContext object.
    """
    
    @abstractmethod
    async def analyze(
        self, 
        context: AIContext
    ) -> RootCauseResult:
        pass

    @abstractmethod
    async def explain(self, analysis_id: str) -> Dict[str, Any]:
        pass

