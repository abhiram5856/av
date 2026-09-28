import os
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL_DIR = os.path.join(REPO_ROOT, "backend", "data", "external_field")

MAPPINGS = {
    "tomato_pakistan": {
        "Tomato_Early_blight": "tomato_early_blight",
        "Tomato_leaf_late_blight": "tomato_late_blight",
        "Tomato_septora_leaf_spot": "tomato_septoria_leaf_spot",
        "Tomato_mold_leaf": "tomato_leaf_mold",
        "Tomato_leaf_yellow_curl_virus": "tomato_yellow_leaf_curl_virus",
        "Tomato_Healthy": "tomato_healthy"
    },
    "potato_pldd_up": {
        "EB": "potato_early_blight",
        "LB": "potato_late_blight",
        "Healthy": "potato_healthy"
    },
    "groundnut_karnataka": {
        "early_leaf_spot": "groundnut_early_leaf_spot",
        "healthy leaf": "groundnut_healthy",
        "late leaf spot": "groundnut_late_leaf_spot",
        "rust": "groundnut_rust"
    },
    "chilli_india": {
        "Healthy Leaf": "chilli_healthy",
        "Curl Virus": "chilli_leaf_curl"
    },
    "maize_field": {}
}

def is_derived(filename):
    lower_f = filename.lower()
    return any(x in lower_f for x in ["aug", "flip", "rot", "_0_", "_1_", "copy"])

def main():
    originals = []
    derived = []
    
    for dataset, mapping in MAPPINGS.items():
        ds_dir = os.path.join(EXTERNAL_DIR, dataset)
        if not os.path.exists(ds_dir): continue
        
        for root, dirs, files in os.walk(ds_dir):
            for f in files:
                if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                
                folder_name = os.path.basename(root)
                mapped = None
                for k, v in mapping.items():
                    if k.lower() in folder_name.lower():
                        mapped = v
                        break
                        
                if not mapped: continue
                if dataset == "groundnut_karnataka" and "Raw_Data" not in root: continue
                
                path = os.path.join(root, f)
                entry = {
                    "path": path,
                    "canonical_class": mapped,
                    "source_dataset": dataset,
                    "original_or_derived": "derived" if is_derived(f) else "original"
                }
                
                if is_derived(f):
                    derived.append(entry)
                else:
                    originals.append(entry)

    with open(os.path.join(REPO_ROOT, "evaluation", "external_original_field_candidates.json"), "w") as f:
        json.dump(originals, f, indent=2)
        
    with open(os.path.join(REPO_ROOT, "evaluation", "external_derived_field_candidates.json"), "w") as f:
        json.dump(derived, f, indent=2)

    print(f"Originals saved: {len(originals)}")
    print(f"Derived saved: {len(derived)}")
    
    # Calculate coverage
    coverage_orig = {}
    coverage_der = {}
    for x in originals: coverage_orig[x['canonical_class']] = coverage_orig.get(x['canonical_class'], 0) + 1
    for x in derived: coverage_der[x['canonical_class']] = coverage_der.get(x['canonical_class'], 0) + 1
    
    print("\nORIGINAL COVERAGE:")
    for k,v in sorted(coverage_orig.items()): print(f"{k}: {v}")
    
    print("\nDERIVED COVERAGE:")
    for k,v in sorted(coverage_der.items()): print(f"{k}: {v}")

if __name__ == '__main__':
    main()
