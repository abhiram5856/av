"""
AgriVision AI — Multimodal Decision Scorer
=================================================
A deterministic decision-support indicator separating model certainty from action priority.
It evaluates visual evidence, environmental compatibility, and growth-stage context.
It is NOT a biological ground-truth severity probability.
"""

from typing import Dict, Any, List

class MultimodalConcernScorer:
    def __init__(self):
        # Weights for action priority
        self.W_VISUAL = 0.50
        self.W_ENV = 0.30
        self.W_GROWTH = 0.20

    def calculate_concern(self, 
                          disease_name: str,
                          is_healthy: bool,
                          ml_confidence: float, 
                          lesion_area_ratio: float, 
                          environmental_evidence: dict,
                          growth_stage: str,
                          image_quality_status: str) -> dict:
        """
        Calculate the transparent Decision Support Score.
        """
        score = 0.0
        contributing_factors = []
        limiting_factors = []
        environmental_conflict = False
        evidence_quality = "Good"

        # 1. Abstention / Uncertainty Handling
        # If confidence is low or image quality is poor, flag as uncertain.
        is_uncertain = False
        if ml_confidence < 0.60:
            is_uncertain = True
            limiting_factors.append(f"Model confidence is too low ({ml_confidence*100:.1f}%) for a definitive diagnosis.")
        if image_quality_status != "acceptable":
            is_uncertain = True
            evidence_quality = "Poor"
            limiting_factors.append("Poor image quality. Diagnosis may be unreliable.")

        if is_uncertain:
            return {
                "score": 0.0,
                "level": "Uncertain - Seek Expert Confirmation",
                "evidence_quality": evidence_quality,
                "environmental_conflict": False,
                "contributing_factors": contributing_factors,
                "limiting_factors": limiting_factors,
                "visual_evidence_level": "Unknown",
                "growth_stage_vulnerability": "Unknown"
            }

        # If model confidently predicts healthy, no action priority needed
        if is_healthy:
            contributing_factors.append(f"Model confidently predicts healthy crop ({ml_confidence*100:.1f}%).")
            return {
                "score": 0.0,
                "level": "Healthy - Monitor Regularly",
                "evidence_quality": evidence_quality,
                "environmental_conflict": False,
                "contributing_factors": contributing_factors,
                "limiting_factors": limiting_factors,
                "visual_evidence_level": "Minimal",
                "growth_stage_vulnerability": "Normal"
            }

        # 2. Visual Evidence (Grad-CAM) (max 50)
        visual_pts = 0.0
        if lesion_area_ratio >= 0.3:
            visual_pts = 50.0
            visual_desc = "Strong"
        elif lesion_area_ratio >= 0.1:
            visual_pts = 30.0
            visual_desc = "Moderate"
        else:
            visual_pts = 10.0
            visual_desc = "Weak"

        score += visual_pts
        contributing_factors.append(f"Visual evidence is {visual_desc} (lesion ratio: {lesion_area_ratio:.2f}) adding {visual_pts:.1f} pts.")

        # 3. Environmental Compatibility (max 30)
        env_status = environmental_evidence.get("overall_compatibility", "Insufficient Evidence")
        env_pts = 0.0
        if env_status == "Strongly Compatible":
            env_pts = 30.0
            contributing_factors.append("Environmental conditions strongly support disease spread (+30 pts).")
        elif env_status == "Compatible":
            env_pts = 20.0
            contributing_factors.append("Environmental conditions are compatible with disease (+20 pts).")
        elif env_status == "Partially Compatible":
            env_pts = 10.0
            contributing_factors.append("Environmental conditions are partially compatible (+10 pts).")
        elif env_status == "Incompatible":
            env_pts = 0.0
            environmental_conflict = True
            limiting_factors.append("Environmental conditions conflict with predicted disease habitat.")
        else:
            env_pts = 0.0
            evidence_quality = "Insufficient"
            limiting_factors.append("Insufficient environmental evidence available.")

        score += env_pts

        # 4. Growth-Stage Vulnerability (max 20)
        growth_pts = 0.0
        gs = (growth_stage or "").title()
        if gs in ["Seedling", "Flowering"]:
            growth_pts = 20.0
            vulnerability = "Higher Vulnerability"
            contributing_factors.append(f"Growth stage '{gs}' is highly vulnerable (+20 pts).")
        elif gs in ["Vegetative", "Fruiting"]:
            growth_pts = 10.0
            vulnerability = "Normal Vulnerability"
            contributing_factors.append(f"Growth stage '{gs}' has normal vulnerability (+10 pts).")
        elif gs == "Harvest":
            growth_pts = 5.0
            vulnerability = "Lower Vulnerability"
            contributing_factors.append(f"Growth stage '{gs}' has lower vulnerability (+5 pts).")
        else:
            vulnerability = "Unknown"
            limiting_factors.append("Growth stage is unknown.")
        
        score += growth_pts

        score = min(max(score, 0.0), 100.0)
        
        # 5. Action Priority Level
        if evidence_quality in ["Poor", "Insufficient"] and env_status == "Insufficient Evidence":
            level = "Moderate Priority (Limited Data)"
        elif score >= 75:
            level = "Critical Action Required"
        elif score >= 50:
            level = "High Priority"
        elif score >= 25:
            level = "Moderate Priority"
        else:
            level = "Low Priority"

        return {
            "score": round(score, 1),
            "level": level,
            "evidence_quality": evidence_quality,
            "environmental_conflict": environmental_conflict,
            "contributing_factors": contributing_factors,
            "limiting_factors": limiting_factors,
            "visual_evidence_level": visual_desc,
            "growth_stage_vulnerability": vulnerability
        }
