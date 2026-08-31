# Rule-Based Logic Engine & Knowledge Base Guide

To power the `AgronomicLogicEngine` (as mentioned in the `BACKEND_DOCS.md`), your backend needs a structured knowledge base of agronomic rules. This ensures that the neural network's visual predictions are physically and biologically sound based on environmental data.

## 1. The Rules JSON (Knowledge Base)

We have created an initial rule set in `knowledge_base/agronomic_rules.json`. This file acts as the "Symbolic" part of your Neuro-Symbolic AI.

Here is the structure:
```json
{
  "Tomato_Late_Blight": {
    "visual_concepts_required": ["water_soaked_spots", "necrotic_lesions"],
    "environmental_conditions": {
      "min_humidity": 0.85,
      "leaf_wetness_required": true
    },
    "severity_multiplier": 1.2,
    "logic_rule_text": "If Tomato_Late_Blight THEN (Water Soaked Spots AND Necrotic Lesions AND High Humidity)"
  }
}
```

## 2. Implementing the AgronomicLogicEngine

Your backend teammates should implement a Python class that loads this JSON file and evaluates the neural network's output against it.

```python
# backend/models/logic_engine.py
import json

class AgronomicLogicEngine:
    def __init__(self, rules_path="knowledge_base/agronomic_rules.json"):
        with open(rules_path, "r") as f:
            self.rules = json.load(f)

    def evaluate_rule(self, disease_hypothesis, visual_probs, env_data):
        """
        disease_hypothesis: The disease predicted by the Neural Network
        visual_probs: Dictionary of visual concepts detected (e.g. {"water_soaked_spots": 0.9})
        env_data: Dictionary of sensor data (e.g. {"humidity": 0.88, "leaf_wetness": True})
        """
        if disease_hypothesis not in self.rules:
            # If no rule exists, fallback to neural network confidence
            return 1.0, "No agronomic rule applied."
            
        rule = self.rules[disease_hypothesis]
        
        # 1. Check Environmental Constraints
        env_score = 1.0
        if env_data.get("humidity", 0) < rule["environmental_conditions"]["min_humidity"]:
            env_score *= 0.5 # Penalty for low humidity when high is required
            
        # 2. Check Visual Concepts
        visual_score = 0.0
        for concept in rule["visual_concepts_required"]:
            visual_score += visual_probs.get(concept, 0.0)
        visual_score /= len(rule["visual_concepts_required"])
        
        # Calculate final satisfaction score (Simple neuro-symbolic fusion)
        satisfaction_score = (env_score * 0.4) + (visual_score * 0.6)
        
        return satisfaction_score, rule["logic_rule_text"]
```

## 3. How it Connects to the API
When the `/diagnose` endpoint runs, it will pass the extracted visual concepts and the farmer's manual humidity inputs into this `AgronomicLogicEngine`. If the `satisfaction_score` is high (e.g., > 0.8), the diagnosis is confirmed as "physically sound" and returned to the frontend.
