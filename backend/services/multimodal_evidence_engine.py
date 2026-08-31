"""
Multimodal Evidence Engine for AgriVision AI
============================================
Evaluates environmental and domain context to cross-reference visual diagnosis.
This does NOT modify the visual prediction probabilities. It provides a separate
evidence assessment layer for the final recommendation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

class MultimodalEvidenceEngine:
    def __init__(self, knowledge_base_path: str):
        self.kb_path = Path(knowledge_base_path)
        self.rules = self._load_rules()
        
    def _load_rules(self) -> dict:
        try:
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {}

    def evaluate(self, 
                 disease_name: str, 
                 temperature: float = None, 
                 humidity: float = None, 
                 ph: float = None,
                 soil_moisture: float = None) -> Dict[str, Any]:
        """
        Evaluate environmental readings against disease knowledge base.
        Returns structured evidence labels.
        """
        if disease_name not in self.rules:
            return {
                "overall_compatibility": "Insufficient Evidence",
                "factors": [{"factor": "knowledge_base", "compatibility": "Unknown", "explanation": "Disease not in KB."}]
            }
            
        rule = self.rules[disease_name]
        env_reqs = rule.get("environmental_conditions", {})
        
        factors = []
        score = 0
        total_factors = 0
        
        # Temperature Check
        if temperature is not None and "temp_range_c" in env_reqs:
            t_min, t_max = env_reqs["temp_range_c"]
            total_factors += 1
            if t_min <= temperature <= t_max:
                factors.append({"factor": "temperature", "compatibility": "Supported", "explanation": f"{temperature}°C is within favorable range ({t_min}-{t_max}°C)."})
                score += 1
            elif abs(temperature - t_min) <= 5 or abs(temperature - t_max) <= 5:
                factors.append({"factor": "temperature", "compatibility": "Partially Supported", "explanation": f"{temperature}°C is marginally favorable."})
                score += 0.5
            else:
                factors.append({"factor": "temperature", "compatibility": "Environmental Mismatch", "explanation": f"{temperature}°C is outside favorable range ({t_min}-{t_max}°C)."})
                score -= 1

        # Humidity Check
        if humidity is not None and "min_humidity" in env_reqs:
            total_factors += 1
            min_h = env_reqs["min_humidity"] * 100
            if humidity >= min_h:
                factors.append({"factor": "humidity", "compatibility": "Supported", "explanation": f"{humidity}% humidity supports disease spread (needs >{min_h}%)."})
                score += 1
            elif humidity >= min_h - 15:
                factors.append({"factor": "humidity", "compatibility": "Partially Supported", "explanation": f"Humidity is moderately favorable."})
                score += 0.5
            else:
                factors.append({"factor": "humidity", "compatibility": "Environmental Mismatch", "explanation": f"{humidity}% is too dry for rapid spread."})
                score -= 1

        # Determine Overall Compatibility
        if total_factors == 0:
            overall = "Insufficient Evidence"
        else:
            avg_score = score / total_factors
            if avg_score >= 0.8:
                overall = "Strongly Compatible"
            elif avg_score >= 0.4:
                overall = "Compatible"
            elif avg_score >= 0.0:
                overall = "Partially Compatible"
            else:
                overall = "Incompatible"
                
        return {
            "overall_compatibility": overall,
            "factors": factors
        }
