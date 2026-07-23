class SeverityScoringEngine:
    """
    NOVA Custom Disease Severity Scoring Engine.
    
    This engine calculates a final 'Severity Risk Score' (0-100) by combining:
    1. Visual/ML Evidence (Grad-CAM lesion area and Model Confidence)
    2. Environmental Factors (Humidity, Temperature)
    3. Soil Health (pH levels)
    """
    
    def __init__(self):
        # Disease categories to apply specific environmental multipliers
        self.fungal_diseases = ['blight', 'mold', 'mildew', 'rust', 'scab', 'spot', 'rot']
        self.viral_diseases = ['virus', 'curl']
        
    def _is_fungal(self, disease_name: str) -> bool:
        disease_name = disease_name.lower()
        return any(fungal_type in disease_name for fungal_type in self.fungal_diseases)

    def _calculate_visual_base_score(self, ml_confidence: float, lesion_area_ratio: float) -> float:
        """
        Base score derived strictly from the computer vision model.
        lesion_area_ratio: 0.0 to 1.0 (calculated via Grad-CAM mask density)
        ml_confidence: 0.0 to 1.0
        """
        # We weigh the physical lesion area higher than just the model confidence
        # Area provides scale of damage, confidence provides certainty.
        base_score = (lesion_area_ratio * 0.7) + (ml_confidence * lesion_area_ratio * 0.3)
        return base_score * 100 # Convert to 0-100 scale

    def _calculate_environmental_multiplier(self, disease_name: str, humidity: float, temp_c: float) -> float:
        """
        Adjusts the severity based on weather conditions.
        Fungal diseases spread exponentially in high humidity and moderate temperatures.
        """
        multiplier = 1.0
        
        if self._is_fungal(disease_name):
            # Fungal diseases love high humidity (>80%) and temps between 20-30C
            if humidity >= 80:
                multiplier += 0.25
            elif humidity >= 60:
                multiplier += 0.10
                
            if 20 <= temp_c <= 30:
                multiplier += 0.15
        else:
            # For general pests or viruses, extreme weather stresses the plant further
            if temp_c > 35 or temp_c < 10:
                multiplier += 0.15
            if humidity < 30: # Dry conditions stress plants
                multiplier += 0.10
                
        return multiplier

    def _calculate_soil_stress_multiplier(self, ph: float) -> float:
        """
        Plants in highly acidic or highly alkaline soil are stressed and more susceptible 
        to rapid disease spread. Optimal pH for most crops is 6.0 - 7.5.
        """
        multiplier = 1.0
        
        if ph < 5.5: # Highly acidic
            multiplier += 0.20
        elif ph > 7.5: # Highly alkaline
            multiplier += 0.20
        elif ph < 6.0: # Slightly acidic
            multiplier += 0.10
            
        return multiplier

    def calculate_final_severity(self, 
                                 disease_name: str, 
                                 ml_confidence: float, 
                                 lesion_area_ratio: float, 
                                 humidity: float, 
                                 temp_c: float, 
                                 ph_level: float) -> dict:
        """
        Returns the final calculated severity score and a human-readable categorization.
        """
        # 1. Base Score from Computer Vision
        base_score = self._calculate_visual_base_score(ml_confidence, lesion_area_ratio)
        
        # 2. Apply Multipliers
        env_multiplier = self._calculate_environmental_multiplier(disease_name, humidity, temp_c)
        soil_multiplier = self._calculate_soil_stress_multiplier(ph_level)
        
        # 3. Final Calculation (Capped at 100)
        final_score = base_score * env_multiplier * soil_multiplier
        final_score = min(max(final_score, 0.0), 100.0)
        
        # 4. Categorization
        if final_score < 30:
            category = "Mild (Monitor closely)"
            urgency = "Low"
        elif final_score < 70:
            category = "Moderate (Treatment recommended)"
            urgency = "Medium"
        else:
            category = "Severe (Immediate action required)"
            urgency = "High"
            
        return {
            "disease": disease_name,
            "base_vision_score": round(base_score, 2),
            "environmental_risk_factor": round(env_multiplier, 2),
            "soil_stress_factor": round(soil_multiplier, 2),
            "final_severity_score": round(final_score, 2),
            "severity_category": category,
            "urgency": urgency
        }

# --- Example Usage ---
if __name__ == "__main__":
    engine = SeverityScoringEngine()
    
    # Scenario: Tomato Late Blight (Fungal)
    # ML is 95% confident, Grad-CAM shows 40% of the leaf is covered in lesions.
    # Weather is highly conducive to fungal spread (85% humidity, 24C).
    # Soil is slightly acidic (5.2 pH) adding stress.
    
    result = engine.calculate_final_severity(
        disease_name="tomato_late_blight",
        ml_confidence=0.95,
        lesion_area_ratio=0.40,
        humidity=85.0,
        temp_c=24.0,
        ph_level=5.2
    )
    
    import json
    print(json.dumps(result, indent=4))
