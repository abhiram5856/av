# AgriVision AI — Scientific Claims Audit

**Date:** 2026-08-26  
**Purpose:** Document which project claims are scientifically defensible, which are heuristic, which are simulated, and which are unverified.

---

## Claim Classification

### VERIFIED — Actual ML

| Claim | Evidence |
|-------|---------|
| MobileNetV3-Small disease classifier | Gate 0 verification — 24-class checkpoint confirmed |
| Grad-CAM explainability | `backend/models/explainability.py` — genuine PyTorch forward/backward hooks |
| 5-view Test-Time Augmentation (TTA) | `diagnose.py` — 5 transforms, probability averaging before argmax |
| Domain adaptation (backbone frozen, head fine-tuned) | `evaluate_field_final.py` — optimizer targets only classifier parameters |
| FAISS vector retrieval (RAG) | `backend/vectorstore/faiss_store.py` — genuine IndexFlatL2 search |
| Sentence embedding (RAG) | `backend/embeddings/local_embedder.py` — `all-MiniLM-L6-v2` |

---

### HEURISTIC — Deterministic Rule-Based Logic

These components are NOT trained ML. They are deterministic rule evaluation systems.

| Component | Honest Label | What It Actually Does |
|-----------|-------------|----------------------|
| Environmental Risk Assessment | **Heuristic Risk Estimate** | Rules based on temperature, humidity thresholds from `agronomic_rules.json` |
| Evidence Consistency Engine (TRACE-RCE v3) | **Heuristic Evidence Assessment** | Checks if visual prediction and environmental readings agree; not causal inference |
| Severity Scoring Engine | **Heuristic Severity Estimate** | Weighted formula using Grad-CAM lesion ratio, confidence, and environmental factors |
| Grad-CAM lesion_area_ratio | **Attention-Based Approximation** | Pixel thresholding on Grad-CAM heatmap — NOT biological ground-truth lesion measurement |

> **Correct wording:** "The Evidence Consistency Engine is a rule-based system that checks whether environmental context supports the visual diagnosis."  
> **Incorrect wording:** "Causal AI" / "Root cause inference" / "Trained predictive model"

---

### SIMULATED — Clearly Labelled

| Component | Status | Frontend Label |
|-----------|--------|---------------|
| IoT Sensor Telemetry | Simulated in-memory data | "Simulated Data" badge displayed in UI |
| Environmental readings (when no OpenWeather key) | Static defaults | Logged as `OPENWEATHER_API_KEY not set. Using default static weather values.` |

**IoT architecture exists and is correct** — the pathway from ESP32 → API → Database → Dashboard is implemented. Current implementation uses simulated data pending real hardware deployment.

---

### UNVERIFIED — Cannot Be Reproduced

| Claim | Status | Reason |
|-------|--------|--------|
| 91.75% laboratory accuracy | **UNVERIFIED** | No raw prediction artifact. No confusion matrix from this run. Cannot recompute. |
| 73.03% post-adaptation field accuracy | **UNVERIFIED** | `nova_mobilenet_v3_field_tuned.pth` exists but was not re-evaluated in this audit. |
| Top-3 accuracy: 99.86% | **UNVERIFIED** | Same as 91.75% — no artifact. |
| Dataset provenance (field images) | **PARTIALLY UNVERIFIED** | No geographic metadata confirming Telangana origin. |

---

## Language to Use / Avoid

### Use

- "Heuristic environmental risk estimate based on temperature and humidity thresholds"
- "Rule-based evidence assessment that checks for agreement between visual diagnosis and environmental context"
- "MobileNetV3-Small image classifier with 24-class output"
- "Grad-CAM visual attention heatmap"
- "Approximate lesion coverage derived from Grad-CAM attention thresholding"
- "Field-domain adaptation experiment — classifier head fine-tuning with frozen backbone"
- "Zero-shot field accuracy of 33.06% consistent with documented lab-to-field domain gap"

### Avoid

- ~~Causal AI~~
- ~~Causal inference~~
- ~~Root cause proof~~
- ~~Precision agriculture~~ (IoT is simulated)
- ~~Real-time Telangana farm validation~~
- ~~91.75% accuracy~~ (unverified)
- ~~Field accuracy of 73.03%~~ (requires re-evaluation)

---

## Safe Viva Statements

**On the classifier:**  
*"We trained a MobileNetV3-Small classifier on the PlantVillage dataset, achieving competitive performance in controlled laboratory conditions. On real-world field images, zero-shot accuracy dropped to 33.06%, confirming the well-documented lab-to-field domain gap."*

**On domain adaptation:**  
*"We conducted a field domain adaptation experiment where the backbone was frozen and only the classifier head was fine-tuned on field-domain images. A script to reproduce this evaluation is available at `backend/scripts/evaluate_field_final.py`."*

**On the evidence engine:**  
*"The TRACE Evidence Consistency Engine is a deterministic rule-based system. It checks whether the predicted disease class is environmentally plausible given current temperature and humidity readings. It does not perform causal inference."*

**On RAG:**  
*"The chatbot uses retrieval-augmented generation. Farmer queries are embedded using `all-MiniLM-L6-v2`, semantically similar agronomic rules are retrieved from a FAISS index, and a grounded prompt is constructed before calling the Groq LLM API. Every response includes a debug_retrieval field with chunk IDs and similarity scores."*
