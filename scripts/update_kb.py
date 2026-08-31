import json
from pathlib import Path

kb_path = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\knowledge_base\agronomic_rules.json")

with open(kb_path, 'r') as f:
    kb = json.load(f)

new_entries = {
    # Cotton
    "cotton_bacterial_blight": {
        "visual_concepts_required": ["angular_water_soaked_spots", "vein_blight", "black_arm"],
        "environmental_conditions": {"min_humidity": 0.85, "temp_range_c": [30, 35]},
        "telangana_relevance": "Direct Evidence: High temperatures and monsoon humidity in Telangana favor Xanthomonas citri pv. malvacearum.",
        "treatment": "Apply Copper oxychloride or Streptomycin sulfate. Use resistant varieties.",
        "logic_rule_text": "If cotton_bacterial_blight THEN (Angular spots AND High Humidity AND High Temps)"
    },
    "cotton_grey_mildew": {
        "visual_concepts_required": ["white_grey_powdery_growth", "angular_pale_spots", "defoliation"],
        "environmental_conditions": {"min_humidity": 0.75, "temp_range_c": [20, 30]},
        "telangana_relevance": "Direct Evidence: Common in central and southern regions during cool, humid periods.",
        "treatment": "Apply Carbendazim or Wettable sulfur. Ensure proper spacing for aeration.",
        "logic_rule_text": "If cotton_grey_mildew THEN (Grey powdery growth AND Moderate temps AND High humidity)"
    },
    "cotton_leaf_curl_virus": {
        "visual_concepts_required": ["upward_downward_leaf_curl", "vein_thickening", "enations"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [25, 35]},
        "telangana_relevance": "Indirect Agronomic Context: Spread by whiteflies (Bemisia tabaci) which thrive in warm, dry spells.",
        "treatment": "No cure. Control whitefly vectors with Imidacloprid. Uproot infected plants.",
        "logic_rule_text": "If cotton_leaf_curl_virus THEN (Leaf curl AND vein thickening -- virus vector)"
    },
    "cotton_healthy": {
        "visual_concepts_required": ["healthy_tissue", "normal_veins"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [21, 37]},
        "telangana_relevance": "Healthy cotton leaf. Monitor for pests and diseases regularly.",
        "treatment": "Maintain standard agronomic practices.",
        "logic_rule_text": "If cotton_healthy THEN (No visible symptoms)"
    },
    
    # Groundnut
    "groundnut_early_leaf_spot": {
        "visual_concepts_required": ["circular_brown_spots", "yellow_halo"],
        "environmental_conditions": {"min_humidity": 0.80, "temp_range_c": [25, 30]},
        "telangana_relevance": "Direct Evidence: Common in Kharif season across Telangana groundnut belts.",
        "treatment": "Apply Chlorothalonil or Mancozeb. Implement crop rotation.",
        "logic_rule_text": "If early_leaf_spot THEN (Brown spots with yellow halos AND High humidity)"
    },
    "groundnut_late_leaf_spot": {
        "visual_concepts_required": ["dark_brown_black_spots", "lower_leaf_surface_spores", "defoliation"],
        "environmental_conditions": {"min_humidity": 0.85, "temp_range_c": [20, 26]},
        "telangana_relevance": "Direct Evidence: Often co-occurs with rust late in the season.",
        "treatment": "Apply Tebuconazole or Hexaconazole. Avoid continuous groundnut cultivation.",
        "logic_rule_text": "If late_leaf_spot THEN (Dark spots mostly underneath AND High humidity)"
    },
    "groundnut_rust": {
        "visual_concepts_required": ["orange_red_pustules", "lower_leaf_surface"],
        "environmental_conditions": {"min_humidity": 0.85, "temp_range_c": [20, 25]},
        "telangana_relevance": "Direct Evidence: Major yield reducer in Telangana when combined with late leaf spot.",
        "treatment": "Apply Chlorothalonil or Triazole fungicides.",
        "logic_rule_text": "If rust THEN (Orange pustules AND High humidity AND Cool temps)"
    },
    "groundnut_healthy": {
        "visual_concepts_required": ["healthy_tissue"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [25, 30]},
        "telangana_relevance": "Healthy groundnut leaf.",
        "treatment": "Standard nutrient management.",
        "logic_rule_text": "If healthy THEN (No visible symptoms)"
    },
    
    # Chilli
    "chilli_leaf_curl": {
        "visual_concepts_required": ["upward_curling", "crumpled_leaves", "stunted_growth"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [25, 35]},
        "telangana_relevance": "Indirect Agronomic Context: Spread by thrips/mites. Thrives in dry, warm conditions common in Khammam/Warangal.",
        "treatment": "Control vectors (thrips/mites) using Spinosad or Abamectin. Destroy infected plants.",
        "logic_rule_text": "If chilli_leaf_curl THEN (Curled leaves AND vector presence)"
    },
    "chilli_leaf_spot": {
        "visual_concepts_required": ["circular_spots", "grey_centers", "dark_margins"],
        "environmental_conditions": {"min_humidity": 0.80, "temp_range_c": [22, 28]},
        "telangana_relevance": "Direct Evidence: Fungal pathogen (Cercospora) favors high humidity.",
        "treatment": "Apply Copper oxychloride or Mancozeb.",
        "logic_rule_text": "If leaf_spot THEN (Circular spots with grey centers AND High humidity)"
    },
    "chilli_whitefly": {
        "visual_concepts_required": ["tiny_white_insects", "sooty_mold", "yellowing"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [25, 35]},
        "telangana_relevance": "Indirect Agronomic Context: Major vector for leaf curl viruses in Telangana.",
        "treatment": "Apply Neem oil or systemic insecticides like Imidacloprid.",
        "logic_rule_text": "If whitefly THEN (Visible whiteflies AND warm weather)"
    },
    "chilli_yellowish": {
        "visual_concepts_required": ["yellowing", "chlorosis"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [20, 35]},
        "telangana_relevance": "Indirect Agronomic Context: Often indicates nutrient deficiency, waterlogging, or early viral infection.",
        "treatment": "Check soil moisture (avoid waterlogging) and apply micronutrients (Zinc/Iron).",
        "logic_rule_text": "If yellowish THEN (Yellowing leaves AND potential stress factors)"
    },
    "chilli_healthy": {
        "visual_concepts_required": ["dark_green_tissue", "flat_leaves"],
        "environmental_conditions": {"min_humidity": 0.0, "temp_range_c": [20, 30]},
        "telangana_relevance": "Healthy chilli plant.",
        "treatment": "Continue regular monitoring.",
        "logic_rule_text": "If healthy THEN (No symptoms)"
    }
}

kb.update(new_entries)

with open(kb_path, 'w') as f:
    json.dump(kb, f, indent=2)
    
print(f"Added {len(new_entries)} entries to knowledge base. Total entries: {len(kb)}")
