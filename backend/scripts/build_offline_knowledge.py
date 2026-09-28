import json
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
knowledge_path = os.path.join(repo_root, "backend", "data", "knowledge_base.json")
output_path = os.path.join(repo_root, "frontend", "public", "data", "offline_knowledge.json")

def build_offline_knowledge():
    with open(knowledge_path, "r") as f:
        kb = json.load(f)
        
    offline_kb = {}
    diseases = kb.get("diseases", {})
    for class_id, entry in diseases.items():
        offline_kb[class_id] = {
            "disease_type": entry.get("disease_type", "Unknown"),
            "symptoms": entry.get("leaf_symptoms", "Unavailable"),
            "prevention": entry.get("prevention", "Unavailable"),
            "chemical_control": entry.get("chemical_control", "Unavailable"),
            "treatment": entry.get("general_management_strategy", "Unavailable"),
            "distinctive_pattern": entry.get("distinctive_symptom_pattern", "Unavailable"),
            "expert_review": "Consult local agronomist if symptoms persist."
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(offline_kb, f, indent=2)
    print(f"Offline knowledge saved to {output_path}")

if __name__ == "__main__":
    build_offline_knowledge()
