# AgriVision AI — Telangana Field Data Audit

**Status:** Complete — Based on actual data inspection  
**Date:** 2026-08-26  
**Auditor:** Independent automated verification

---

## 1. Separation of Dataset Types

The project uses two distinct dataset types. These must never be conflated.

| Type | Dataset | Images | Classes | Domain |
|------|---------|--------|---------|--------|
| **Laboratory (Benchmark)** | PlantVillage (preprocessed) | ~5,667 (all classes) | 24 (active scope) | Controlled white/black segmented backgrounds |
| **Field Domain** | Processed field dataset | 849+ | 24 (active) + 9 out-of-scope | Natural field conditions, variable backgrounds |

---

## 2. Laboratory Benchmark Dataset

**Source:** PlantVillage dataset (publicly available)  
**Processing:** White/black background segmentation — NOT real farm photographs  
**Classes:** 24 active Telangana-scoped classes (post-Cotton removal)  
**Limitation:**  

> PlantVillage images are taken under controlled laboratory conditions with uniform backgrounds. Performance on this dataset does NOT predict real-world farm performance. The documented lab-to-field domain gap (zero-shot field accuracy: 33.06% vs claimed lab baseline) confirms this limitation.

**Split (from `statistics.json`):**  

| Split | Images |
|-------|--------|
| Train | 4,088 |
| Validation | 1,021 |
| Test | 558 |
| **Total** | **5,667** |

**Leakage check (per `statistics.json`):**

| Check | Result |
|-------|--------|
| Train-Test overlap | **0** |
| Val-Test overlap | **0** |
| Duplicates | 193 (excluded from splits) |

**Verdict:** No detectable leakage per `statistics.json`. Raw images not present in repository to independently re-verify hashing algorithm, therefore status is **PARTIALLY VERIFIED**.

---

## 3. Real-World Field Domain Dataset

**Location:** `backend/data/processed_field_dataset/`  
**Split file:** `backend/data/field_splits.json`  
**Test images evaluated:** 732 (within 24-class scope)  
**Out-of-scope images skipped:** 117 (classes outside active 24-class registry)  

**Image provenance:**  
Source documentation is **incomplete**. The processed field dataset was generated from raw data in `backend/data/raw_field_data/`. The exact original source (Kaggle, local field photos, scraping, etc.) is not documented in metadata files.

> [!WARNING]
> These images may not exclusively originate from Telangana farms. They represent "field-domain" images (natural backgrounds, variable lighting) as distinct from PlantVillage laboratory images. Do NOT claim statewide Telangana validation unless farm location metadata is attached to images.

**Correct claim:** "Field-domain evaluation on a dataset of real-world crop disease images with natural backgrounds."  
**Incorrect claim:** "Validated on images from Telangana farms."

---

## 4. Domain Adaptation Experiment

**Zero-shot field performance** (verified, Gate 0, 2026-08-26):

| Metric | Value | Notes |
|--------|-------|-------|
| Accuracy | **33.06%** | Lab model on field images, no adaptation |
| Macro F1 | **8.55%** | Low due to class imbalance in field test set |
| Macro Precision | 9.59% | |
| Macro Recall | 13.81% | |
| Images evaluated | 732 | |

**Post-adaptation performance** (from `evaluate_field_final.py` script):  
Claims 73.03% Accuracy and 76.22% Macro F1 after classifier head fine-tuning on field training split. **Status: UNVERIFIED** — the fine-tuned checkpoint `nova_mobilenet_v3_field_tuned.pth` exists but was not re-evaluated during this audit run (no time guarantee that field training split was not contaminated).

**To verify independently:** `python backend/scripts/evaluate_field_final.py`

---

## 5. Safe Claims for Presentation

**Safe to claim:**
- "We evaluated the laboratory-trained model on a held-out field-domain test set and observed a zero-shot accuracy of 33.06%, consistent with the expected lab-to-field domain gap."
- "A domain adaptation experiment was conducted by freezing the MobileNetV3 backbone and fine-tuning only the classifier head on field-domain training images."

**Do NOT claim:**
- "Validated on Telangana farm images" — provenance unverified
- "Field accuracy of 73.03%" — requires re-run from field_tuned checkpoint
- "Statewide Telangana validation" — no geographic metadata
