"""
Knowledge Engine for AgriVision-AI.

Loads disease-specific, source-backed agronomic knowledge from the structured
knowledge_base.json. Does NOT hallucinate or fill gaps with generic content.

Any field returning "NOT_AVAILABLE" or "Specific guidance unavailable" reflects
genuine absence of verified source evidence — not a system error.
"""

import json
import os
from typing import Dict, Optional, Any, List


class KnowledgeEngine:
    # These sentinel values in the DB indicate intentional absence of data
    NOT_AVAILABLE_SENTINELS = {
        "NOT_AVAILABLE",
        "NOT_APPLICABLE",
        "NOT_SPECIFICALLY_APPLICABLE",
    }

    def __init__(self, data_path: str = "backend/data/knowledge_base.json"):
        self.data_path = data_path
        self.knowledge_base: Dict[str, Any] = {}
        self._load_knowledge()

    def _load_knowledge(self):
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.knowledge_base = json.load(f).get("diseases", {})
            except Exception as e:
                print(f"[KnowledgeEngine] Error loading knowledge base: {e}")
                self.knowledge_base = {}
        else:
            print(f"[KnowledgeEngine] Knowledge base not found at {self.data_path}")
            self.knowledge_base = {}

    def _is_available(self, value: Optional[str]) -> bool:
        """Returns False if value is None, empty, or a NOT_AVAILABLE sentinel."""
        if not value:
            return False
        return value.strip() not in self.NOT_AVAILABLE_SENTINELS

    def get_disease_knowledge(self, canonical_class: str) -> Optional[Dict[str, Any]]:
        """Return full disease knowledge dict, or None if class not in DB."""
        return self.knowledge_base.get(canonical_class)

    def get_disease_summary(self, canonical_class: str) -> Dict[str, Any]:
        """
        Return a structured disease summary for the API response.
        Only returns fields that have genuine verified content.
        All missing or NOT_AVAILABLE fields are clearly labeled.
        """
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return {
                "available": False,
                "message": "Specific guidance unavailable — consult local agricultural extension guidance.",
            }

        summary = {
            "available": True,
            "disease_name": kb.get("disease_name"),
            "crop": kb.get("crop"),
            "disease_type": kb.get("disease_type"),
            "pathogen": kb.get("pathogen_or_cause") if self._is_available(kb.get("pathogen_or_cause")) else None,
            "scientific_name": kb.get("scientific_name") if self._is_available(kb.get("scientific_name")) else None,
            "parts_affected": kb.get("primary_plant_parts_affected", []),
            "leaf_symptoms": kb.get("leaf_symptoms") if self._is_available(kb.get("leaf_symptoms")) else None,
            "early_symptoms": kb.get("early_symptoms") if self._is_available(kb.get("early_symptoms")) else None,
            "advanced_symptoms": kb.get("advanced_symptoms") if self._is_available(kb.get("advanced_symptoms")) else None,
            "distinctive_pattern": kb.get("distinctive_symptom_pattern") if self._is_available(kb.get("distinctive_symptom_pattern")) else None,
            "lookalike_conditions": kb.get("lookalike_conditions", []),
            "visual_differentiation": kb.get("visual_differentiation") if self._is_available(kb.get("visual_differentiation")) else None,
            "knowledge_completeness": kb.get("knowledge_completeness", "UNKNOWN"),
        }
        return summary

    def get_prevention_guidance(self, canonical_class: str) -> str:
        """
        Return DISEASE-SPECIFIC prevention guidance.
        Falls back to safe default message only if class not in DB.
        """
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return "Specific prevention guidance unavailable — consult local agricultural extension guidance."

        prevention = kb.get("prevention")
        if self._is_available(prevention):
            return prevention

        return "Specific prevention guidance unavailable for this disease — consult local agricultural extension guidance."

    def get_current_management(self, canonical_class: str) -> Optional[str]:
        """Return disease-specific current management (what to do now)."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return None
        cultural = kb.get("cultural_management")
        sanitation = kb.get("sanitation")
        parts = []
        if self._is_available(cultural):
            parts.append(cultural)
        if self._is_available(sanitation):
            parts.append(sanitation)
        return " ".join(parts) if parts else None

    def get_chemical_control(self, canonical_class: str) -> str:
        """
        Return chemical control guidance.
        NEVER returns a specific dosage or rate unless explicitly sourced.
        """
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return (
                "Use only products currently registered/labeled for this crop and disease "
                "in your region and follow the product label or local agricultural extension guidance."
            )

        chem = kb.get("chemical_control")
        if self._is_available(chem):
            return chem

        return (
            "Use only products currently registered/labeled for this crop and disease "
            "in your region and follow the product label or local agricultural extension guidance."
        )

    def get_environmental_context(self, canonical_class: str) -> Optional[Dict[str, Any]]:
        """Return source-backed environmental context for the disease."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return None
        env = kb.get("favorable_environmental_conditions")
        if not env:
            return None
        # Only return fields that have actual verified content
        result = {}
        for key, value in env.items():
            if self._is_available(value):
                result[key] = value
        return result if result else None

    def get_expert_escalation(self, canonical_class: str) -> str:
        """Return expert escalation conditions."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return "Seek expert confirmation if symptoms spread or additional images remain inconclusive."
        escalation = kb.get("expert_escalation_conditions")
        if self._is_available(escalation):
            return escalation
        return "Seek expert confirmation if symptoms spread or additional images remain inconclusive."

    def get_monitoring_guidance(self, canonical_class: str) -> Optional[str]:
        """Return disease-specific monitoring guidance."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return None
        monitoring = kb.get("monitoring_guidance")
        return monitoring if self._is_available(monitoring) else None

    def get_source_references(self, canonical_class: str) -> List[Dict[str, Any]]:
        """Return source references for the disease."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return []
        return kb.get("source_references", [])

    def get_lookalikes(self, canonical_class: str) -> List[str]:
        """Return visually similar diseases for differential diagnosis."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return []
        return kb.get("lookalike_conditions", [])

    def format_farmer_support(self, canonical_class: str, is_low_confidence: bool = False) -> Dict[str, Any]:
        """
        Return farmer-friendly action guidance.
        ALL text is disease-specific where evidence exists.
        Does NOT fill gaps with generic filler text.
        """
        kb = self.knowledge_base.get(canonical_class)

        if not kb:
            return {
                "what_to_do_now": "Specific guidance unavailable — consult local agricultural extension guidance.",
                "monitor": "Monitor the plant for any symptom progression.",
                "prevention": "Specific prevention guidance unavailable — consult local agricultural extension guidance.",
                "expert_review": "Seek expert confirmation if symptoms spread or additional images remain inconclusive.",
                "rescan_instructions": "Move closer to the affected leaf, use natural light, hold camera steady, focus on one leaf only." if is_low_confidence else None,
                "sources": [],
            }

        disease_type = kb.get("disease_type", "other")

        # What to do NOW — disease-specific, not generic
        what_to_do_now_parts = []
        cultural = kb.get("cultural_management")
        sanitation = kb.get("sanitation")
        if self._is_available(cultural):
            what_to_do_now_parts.append(cultural)
        elif self._is_available(sanitation):
            what_to_do_now_parts.append(sanitation)

        if disease_type == "healthy":
            what_to_do_now = "No disease symptoms detected. Continue routine monitoring and crop maintenance."
        elif what_to_do_now_parts:
            what_to_do_now = what_to_do_now_parts[0]
        else:
            what_to_do_now = "Consult local agricultural extension for specific management guidance."

        # Monitoring — disease-specific
        monitoring = kb.get("monitoring_guidance")
        monitor_text = monitoring if self._is_available(monitoring) else "Monitor the affected plant and nearby plants for symptom progression."

        # Prevention — disease-specific
        prevention = kb.get("prevention")
        prevention_text = prevention if self._is_available(prevention) else "Consult local agricultural extension for specific prevention guidance."

        # Expert review — disease-specific
        expert_review = kb.get("expert_escalation_conditions")
        expert_review_text = expert_review if self._is_available(expert_review) else "Seek expert confirmation if symptoms spread or additional images remain inconclusive."

        # Rescan instructions — only shown if confidence is low
        rescan = (
            "Move closer to the affected leaf. Focus on ONE leaf only. Use natural light (no flash). "
            "Hold camera steady. Avoid shadows. Include the full affected area in frame."
            if is_low_confidence
            else None
        )

        return {
            "what_to_do_now": what_to_do_now,
            "monitor": monitor_text,
            "prevention": prevention_text,
            "expert_review": expert_review_text,
            "rescan_instructions": rescan,
            "sources": self.get_source_references(canonical_class),
        }

    def get_knowledge_completeness_score(self, canonical_class: str) -> str:
        """Return the declared knowledge completeness level for audit purposes."""
        kb = self.knowledge_base.get(canonical_class)
        if not kb:
            return "INSUFFICIENT"
        return kb.get("knowledge_completeness", "UNKNOWN")


# Singleton instance
knowledge_engine = KnowledgeEngine()
