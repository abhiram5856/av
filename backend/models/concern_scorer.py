"""
AgriVision AI — Multimodal Disease Concern Scorer
=================================================
A deterministic decision-support indicator (0-100) combining model confidence, visual evidence, 
environmental compatibility, growth-stage context, and image reliability. 
It is NOT a biological ground-truth severity probability.
"""

from typing import Dict, Any, List

class MultimodalConcernScorer:
    def __init__(self):
        # Weights
        self.W_CONFIDENCE = 0.35
        self.W_VISUAL = 0.25
        self.W_ENV = 0.20
        self.W_GROWTH = 0.10
        self.W_QUALITY = 0.10

    def calculate_concern(self, 
                          disease_name: str,
                          is_healthy: bool,
                          ml_confidence: float, 
                          lesion_area_ratio: float, 
                          environmental_evidence: dict,
                          growth_stage: str,
                          image_quality_status: str) -> dict:
        """
        Calculate the transparent Concern Score.
        """
        score = 0.0
        contributing_factors = []
        limiting_factors = []
        environmental_conflict = False
        evidence_quality = "Good"

        # 1. Model Confidence (max 35)
        # We scale confidence linearly. If healthy, concern drops naturally.
        conf_pts = min(ml_confidence * 35.0, 35.0)
        if is_healthy:
            # If healthy, confidence in health lowers concern. We just zero out the confidence part.
            conf_pts = 0.0
            contributing_factors.append(f"Model predicts healthy crop (confidence: {ml_confidence*100:.1f}%).")
        else:
            score += conf_pts
            contributing_factors.append(f"Model prediction confidence ({ml_confidence*100:.1f}%) contributes {conf_pts:.1f} pts.")

        # 2. Visual Evidence (Grad-CAM) (max 25)
        # Using lesion_area_ratio bounds.
        visual_pts = 0.0
        if lesion_area_ratio >= 0.3:
            visual_pts = 25.0
            visual_desc = "Strong"
        elif lesion_area_ratio >= 0.1:
            visual_pts = 15.0
            visual_desc = "Moderate"
        else:
            visual_pts = 5.0
            visual_desc = "Weak"

        if is_healthy:
            visual_pts = 0.0
            visual_desc = "Minimal"
        else:
            score += visual_pts
            contributing_factors.append(f"Visual evidence is {visual_desc} (lesion ratio: {lesion_area_ratio:.2f}) adding {visual_pts:.1f} pts.")

        # 3. Environmental Compatibility (max 20)
        env_status = environmental_evidence.get("overall_compatibility", "Insufficient Evidence")
        env_pts = 0.0
        if env_status == "Strongly Compatible":
            env_pts = 20.0
            contributing_factors.append("Environmental conditions strongly support disease spread (+20 pts).")
        elif env_status == "Compatible":
            env_pts = 15.0
            contributing_factors.append("Environmental conditions are compatible with disease (+15 pts).")
        elif env_status == "Partially Compatible":
            env_pts = 10.0
            contributing_factors.append("Environmental conditions are partially compatible (+10 pts).")
        elif env_status == "Incompatible":
            env_pts = 0.0
            environmental_conflict = True
            limiting_factors.append("Environmental conditions are incompatible with expected disease habitat.")
        else: # Insufficient Evidence
            env_pts = 0.0
            evidence_quality = "Insufficient"
            limiting_factors.append("Insufficient environmental evidence available.")

        if not is_healthy:
            score += env_pts

        # 4. Growth-Stage Vulnerability (max 10)
        # Basic mapping. Can be extended via knowledge base per disease.
        growth_pts = 0.0
        gs = (growth_stage or "").title()
        if gs in ["Seedling", "Flowering"]:
            growth_pts = 10.0
            vulnerability = "Higher Vulnerability"
            if not is_healthy:
                contributing_factors.append(f"Growth stage '{gs}' is highly vulnerable (+10 pts).")
        elif gs in ["Vegetative", "Fruiting"]:
            growth_pts = 5.0
            vulnerability = "Normal Vulnerability"
            if not is_healthy:
                contributing_factors.append(f"Growth stage '{gs}' has normal vulnerability (+5 pts).")
        elif gs == "Harvest":
            growth_pts = 2.0
            vulnerability = "Lower Vulnerability"
            if not is_healthy:
                contributing_factors.append(f"Growth stage '{gs}' has lower vulnerability (+2 pts).")
        else:
            vulnerability = "Unknown"
            evidence_quality = "Insufficient"
            limiting_factors.append("Growth stage is unknown.")
        
        if not is_healthy:
            score += growth_pts

        # 5. Image Quality (max 10)
        quality_pts = 0.0
        if image_quality_status == "acceptable":
            quality_pts = 10.0
            if not is_healthy:
                contributing_factors.append("Image quality is acceptable (+10 pts).")
        else:
            quality_pts = 0.0
            evidence_quality = "Poor"
            limiting_factors.append("Poor image quality reduces diagnostic reliability.")

        if not is_healthy:
            score += quality_pts

        # If healthy, override score to 0 to be safe.
        if is_healthy:
            score = 0.0
            
        score = min(max(score, 0.0), 100.0)
        
        # 6. Concern Level
        if evidence_quality in ["Poor", "Insufficient"] and env_status == "Insufficient Evidence" and not is_healthy:
            # Only claim insufficient if we are really lacking data and it's diseased
            if ml_confidence < 0.6:
                level = "Insufficient Evidence"
            else:
                level = "Moderate Concern" # Fallback if model is very confident
        elif score >= 75:
            level = "Critical Attention Required"
        elif score >= 50:
            level = "High Concern"
        elif score >= 25:
            level = "Moderate Concern"
        else:
            level = "Low Concern"

        if is_healthy:
            level = "Low Concern"

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
