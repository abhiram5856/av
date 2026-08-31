"""Final independent audit of the active inference path after remediation."""
import sys
sys.path.insert(0, r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.models.class_registry import validate_registry, NUM_CLASSES, CLASS_NAMES, MODEL_CONFIG
import os
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
RESULTS = []

def audit(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((status, label, detail))
    print(f"  [{status}] {label}: {detail}")

print("=" * 65)
print("FINAL INDEPENDENT AUDIT — POST-REMEDIATION")
print("=" * 65)

# 1. Registry integrity
print("\n[1] Class Registry")
try:
    validate_registry()
    audit("validate_registry() passes", True, "no assertion errors")
except AssertionError as e:
    audit("validate_registry() passes", False, str(e))

audit("NUM_CLASSES == 24", NUM_CLASSES == 24, f"NUM_CLASSES={NUM_CLASSES}")
cotton_present = any("cotton" in c for c in CLASS_NAMES)
audit("Cotton absent from active registry", not cotton_present, 
      "cotton_diseased and cotton_healthy removed")

ckpt_fname = MODEL_CONFIG["checkpoint_filename"]
audit("checkpoint_filename points to 24-class file",
      ckpt_fname == "nova_mobilenet_v3_24_classes.pth", ckpt_fname)

# 2. Checkpoint availability
print("\n[2] Checkpoint Availability")
ckpt_path = REPO_ROOT / "backend/models/weights" / ckpt_fname
audit("Active checkpoint exists", ckpt_path.exists(), str(ckpt_path))

legacy_26 = REPO_ROOT / "backend/models/weights/nova_mobilenet_v3_26_classes_init.pth"
audit("26-class checkpoint archived (not in active weights dir)",
      not legacy_26.exists(), "moved to archive/")

archive_26 = REPO_ROOT / "backend/models/weights/archive/nova_mobilenet_v3_26_classes_init.pth"
audit("26-class checkpoint found in archive", archive_26.exists(), str(archive_26))

# 3. Checkpoint loads correctly into 24-class model
print("\n[3] Checkpoint Architecture Verification")
try:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    out_dim = state_dict["classifier.3.weight"].shape[0]
    audit("Checkpoint output dim == 24", out_dim == 24, f"shape={state_dict['classifier.3.weight'].shape}")
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 24)
    model.load_state_dict(state_dict)
    model.eval()
    audit("load_state_dict() into 24-class model succeeds", True, "clean")
except Exception as e:
    audit("Checkpoint loads", False, str(e))

# 4. Knowledge base
print("\n[4] Knowledge Base")
kb_path = REPO_ROOT / "knowledge_base/agronomic_rules.json"
import json
with open(kb_path) as f:
    rules = json.load(f)
audit("Knowledge base covers 24 disease classes", len(rules) == 24, 
      f"entries={len(rules)}")
audit("Apple_Scab removed from knowledge base", "Apple_Scab" not in rules,
      "out-of-scope rule removed")
audit("Tomato late blight present", "tomato_late_blight" in rules, "")
audit("Rice leaf blast present", "rice_leaf_blast" in rules, "")

# 5. FAISS index
print("\n[5] FAISS Vector Store")
faiss_path = REPO_ROOT / "data/faiss_index"
audit("FAISS index file exists", faiss_path.exists(), str(faiss_path))
faiss_pkl = REPO_ROOT / "data/faiss_index.pkl"
audit("FAISS pickle file exists", faiss_pkl.exists(), str(faiss_pkl))

# 6. RAG chatbot
print("\n[6] RAG Chatbot")
chat_path = REPO_ROOT / "backend/api/chat.py"
with open(chat_path) as f:
    chat_src = f.read()
audit("New chat.py has no keyword matching", 
      "if \"paddy\"" not in chat_src and "if 'paddy'" not in chat_src,
      "old mock removed")
audit("New chat.py uses VectorRetriever", "VectorRetriever" in chat_src, "")
audit("New chat.py uses GroqClient", "GroqClient" in chat_src, "")
audit("debug_retrieval field present", "debug_retrieval" in chat_src, "audit trail field")
audit("retrieval_executed field present", "retrieval_executed" in chat_src, "audit field")
audit("Low-confidence advisory present", "LOW_CONF_ADVISORY" in chat_src, "")
audit("Honest fallback (no GROQ key) present", "GROQ_API_KEY not set" in chat_src, "")

# 7. Documentation
print("\n[7] Documentation")
docs = [
    "docs/TELANGANA_FIELD_DATA_AUDIT.md",
    "docs/ML_REPRODUCIBILITY_GUIDE.md",
    "docs/SCIENTIFIC_CLAIMS_AUDIT.md",
    "docs/RAG_ARCHITECTURE.md",
    "evaluation/experiment_metadata.json",
    "evaluation/gate0_checkpoint_verification.json",
]
for doc in docs:
    p = REPO_ROOT / doc
    audit(f"{doc} exists", p.exists(), f"size={p.stat().st_size if p.exists() else 0} bytes")

# 8. Scientific claims check
print("\n[8] Scientific Claims in Metadata")
meta_path = REPO_ROOT / "backend/models/weights/metadata.json"
with open(meta_path) as f:
    meta = json.load(f)
audit("Metadata num_classes == 24", meta["num_classes"] == 24, str(meta["num_classes"]))
audit("91pct claim flagged as UNVERIFIED in metadata",
      "UNVERIFIED" in str(meta.get("evaluation", {})), "")
audit("PlantDoc metric isolated in metadata",
      "DIFFERENT" in str(meta.get("evaluation", {})) or "NOT the" in str(meta.get("evaluation", {})),
      "separated from 24-class baseline")

# Final verdict
print("\n" + "=" * 65)
failures = [r for r in RESULTS if r[0] == "FAIL"]
passes   = [r for r in RESULTS if r[0] == "PASS"]
print(f"FINAL AUDIT: {len(passes)} PASSED, {len(failures)} FAILED")
if not failures:
    print("ALL CHECKS PASSED. Repository is remediation-complete.")
else:
    print("REMAINING FAILURES:")
    for _, label, detail in failures:
        print(f"  - {label}: {detail}")
print("=" * 65)
