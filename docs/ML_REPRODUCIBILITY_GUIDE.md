# AgriVision AI — ML Reproducibility Guide

**Date:** 2026-08-26  
**Purpose:** Provide exact commands to reproduce every verified result. Do not claim reproducibility for experiments that cannot currently be run.

---

## Reproducible Right Now

### 1. Gate 0 — Checkpoint Architecture Verification

Verifies that `nova_mobilenet_v3_24_classes.pth` is architecturally valid and produces 24-class output.

```bash
python scripts/gate0_verify_checkpoint.py
```

**Expected output:**
```
GATE 0 VERDICT: PASS
Checkpoint is verified. Safe to promote to production.
Field zero-shot accuracy: 33.06% on 732 images
```

**Output artifact:** `evaluation/gate0_checkpoint_verification.json`

---

### 2. FAISS Knowledge Base Population

Embeds all 24 disease class rules into FAISS for RAG retrieval.

```bash
python scripts/ingest_knowledge_base.py
```

**Expected output:**
```
Embedding 24 rule(s)...
FAISS index populated and saved successfully.
```

---

### 3. Class Registry Consistency Check

```python
python -c "from backend.models.class_registry import validate_registry, NUM_CLASSES; validate_registry(); print(f'Registry OK: {NUM_CLASSES} classes')"
```

**Expected output:** `Registry OK: 24 classes`

---

### 4. Backend Startup Test

```bash
uvicorn backend.main:app --reload
```

Check: `GET http://localhost:8000/health` returns `{"status": "ok"}`  
Check: Model loading log shows `Classes=24`

---

### 5. RAG Retrieval Test (without LLM)

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "How to treat tomato late blight in Telangana?"}'
```

**Check `retrieval_executed: true` and `debug_retrieval` contains chunk IDs.**

---

## Requires Re-Run (Currently Unverified)

### A. 24-Class Lab Baseline Evaluation

**Status:** UNVERIFIED — requires 24-class held-out PlantVillage test set  
**Script:** Does not exist yet — must be created  
**Blocker:** `evaluation/classification_report.json` contains a 36-class PlantDoc evaluation, not the 24-class model

**To create:**
```python
# Use backend/models/vision_training.py evaluate() function
# Point to the 24-class test set
# Save predictions to evaluation/baseline_predictions.csv
```

### B. Post-Adaptation Field Accuracy (73.03%)

**Status:** UNVERIFIED in this audit  
**Script:** `python backend/scripts/evaluate_field_final.py`  
**Requires:** `nova_mobilenet_v3_field_tuned.pth` + `backend/data/field_splits.json` + field images  
**Note:** If field_splits.json test split was used during training, this result is contaminated. Verify the optimizer parameter list in the fine-tuning script.

---

## Evaluation Artifacts Reference

| File | Content | Status |
|------|---------|--------|
| `evaluation/gate0_checkpoint_verification.json` | Gate 0 full verification report | **VERIFIED** |
| `evaluation/experiment_metadata.json` | Provenance of all cited metrics | **CREATED** |
| `evaluation/metrics.json` | PlantDoc 36-class accuracy (77.6%) | **HISTORICAL — NOT 24-class baseline** |
| `evaluation/classification_report.json` | PlantDoc 36-class per-class report | **HISTORICAL** |
| `evaluation/confusion_matrix.png` | Visual confusion matrix | **HISTORICAL — PlantDoc** |
| `scientific_baseline_results.json` | TRACE-RCE v1/v2 benchmark results | **Different task — not image classification** |
