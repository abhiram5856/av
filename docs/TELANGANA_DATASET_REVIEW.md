# TELANGANA CROP & FIELD DATA DISCOVERY

## STEP 1 — TELANGANA CROP PRIORITIZATION

| Crop | Telangana relevance | Disease data | Field data | Recommended? | Reason |
|------|---------------------|--------------|------------|--------------|--------|
| **Rice** | Very High (Staple) | Yes | Yes | **YES** | Massive Kaggle dataset (Paddy Doctor) provides 10,000+ real Indian field images. |
| **Cotton** | Very High (Major cash crop) | Yes | Yes | **YES** | Key Telangana crop. Some Kaggle Indian field datasets exist, though classes are currently unsupported by baseline. |
| **Maize/Corn** | High | Yes | Yes | **YES** | PlantDoc dataset contains real-world field images matching baseline classes. |
| **Tomato** | High | Yes | Yes | **YES** | PlantDoc dataset contains excellent field data matching our 10 tomato baseline classes. |
| **Chilli** | Very High | Limited | Poor | **NO** | Available datasets are fragmented, unverified, and risk containing synthetic or heavily duplicated images. |
| **Groundnut** | High | Limited | Poor | **NO** | Datasets are largely institutional/gated; insufficient legitimate open-source field data. |

## STEP 2 — DATASET DISCOVERY

1. **Paddy Doctor: Paddy Disease Classification** (Kaggle)
   - Indian field dataset (Manonmaniam Sundaranar University).
2. **PlantDoc Dataset** (GitHub/Roboflow)
   - Web-scraped field/real-world images of Corn and Tomato.
3. **Cotton Leaf Disease Dataset** (Kaggle)
   - Publicly uploaded Indian field cotton datasets.

*(Google Images, synthetic AI, and unverified Kaggle dumps with no provenance were strictly excluded).*

## STEP 3 — DATASET DOCUMENTATION

### Dataset A: Paddy Doctor
- **URL/Source:** Kaggle (MSU, India)
- **License:** Open for Research
- **Crop:** Rice
- **Disease classes:** 9 diseases + 1 healthy
- **Total images:** 10,407
- **Field vs laboratory:** Strictly Field (Indian paddy fields)
- **Geographic relevance:** Very High (India/Telangana)
- **Image resolution:** Varies (mostly smartphone resolutions)
- **Known limitations:** Heavy class imbalance (some classes have <100 images while others have 2000+).

### Dataset B: PlantDoc
- **URL/Source:** GitHub / IIT researchers
- **License:** CC BY 4.0
- **Crop:** Corn, Tomato (among others)
- **Disease classes:** Multiple matching our baseline
- **Total images:** ~2,598
- **Field vs laboratory:** Field/Real-world
- **Geographic relevance:** Medium (Global web scrape, but realistic)
- **Image resolution:** Varies widely
- **Known limitations:** Contains multiple leaves per image; background noise is extremely high.

### Dataset C: Cotton Leaf Disease Dataset
- **URL/Source:** Kaggle
- **License:** CC0 / Public Domain
- **Crop:** Cotton
- **Disease classes:** Bacterial Blight, Curl Virus, etc.
- **Total images:** ~2,000
- **Field vs laboratory:** Field
- **Geographic relevance:** High (India)
- **Known limitations:** Not currently in our 24-class baseline. Domain adaptation would require expanding the classifier head to `NUM_CLASSES = 27`.

## STEP 4 — CLASS MAPPING

Comparing discovered field datasets against `backend/models/class_registry.py` (24 classes):

| Current Baseline Class | Field Dataset Class | Match Status |
|------------------------|---------------------|--------------|
| `rice_leaf_blast` | `blast` (Paddy Doctor) | MATCH |
| `rice_brown_spot` | `brown_spot` (Paddy Doctor) | MATCH |
| `rice_bacterial_leaf_blight` | `bacterial_leaf_blight` (Paddy Doctor) | MATCH |
| `rice_healthy` | `normal` (Paddy Doctor) | MATCH |
| `rice_leaf_scald` | N/A | NO MATCH (Insufficient field data) |
| `rice_sheath_blight` | N/A | NO MATCH (Insufficient field data) |
| `corn_gray_leaf_spot` | `Corn Gray leaf spot` (PlantDoc) | MATCH |
| `corn_leaf_blight` | `Corn leaf blight` (PlantDoc) | MATCH |
| `corn_rust_leaf` | `Corn rust leaf` (PlantDoc) | MATCH |
| `tomato_bacterial_spot` | `Tomato bacterial spot` (PlantDoc) | MATCH |
| `tomato_early_blight` | `Tomato Early blight leaf` (PlantDoc) | MATCH |
| `tomato_late_blight` | `Tomato late blight leaf` (PlantDoc) | MATCH |
| `tomato_leaf_mold` | `Tomato leaf mold` (PlantDoc) | MATCH |
| `tomato_healthy` | `Tomato healthy` (PlantDoc) | MATCH |
| `potato_*` (3 classes) | `Potato_*` (PlantDoc) | MATCH (Optional) |
| `pepper_bell_*` (2 classes)| `Bell_pepper_*` (PlantDoc) | MATCH (Optional) |
| **UNSUPPORTED IN BASELINE** | `Cotton Bacterial Blight` | NEW CLASS REQUIRED |
| **UNSUPPORTED IN BASELINE** | `Cotton Curl Virus` | NEW CLASS REQUIRED |

## STEP 5 — FIELD DATA SUFFICIENCY

- `rice_leaf_blast`: 🟢 Sufficient (>1000 field images)
- `rice_brown_spot`: 🟢 Sufficient (>900 field images)
- `rice_bacterial_leaf_blight`: 🟢 Sufficient (>400 field images)
- `corn_leaf_blight`: 🟡 Limited (~150 field images)
- `tomato_early_blight`: 🟡 Limited (~200 field images)
- `rice_leaf_scald`: 🔴 Insufficient
- `cotton_bacterial_blight`: 🟢 Sufficient (~400 field images)

*No class flagged as 🔴 Insufficient will be included in the final field evaluation.*

## STEP 6 — DOMAIN GAP

**PlantVillage-style images** (our current dataset) were taken in controlled laboratory environments. A single leaf is placed on a homogenous grey/black/white background, photographed under uniform lighting, and cropped.
**Real field images** feature natural soil/sky backgrounds, harsh or dappled sunlight, overlapping leaves, partial occlusion, motion blur from wind, and varying smartphone camera sensors.

**Impact on current 91.75% baseline:**
If we deploy the current model directly to a Telangana farm, accuracy will catastrophically drop (likely into the 40-60% range). The model has likely learned to identify diseases using high-frequency textures that are destroyed by JPEG compression and blur in field conditions, or it may rely on background artifacts that don't exist in the wild.

## STEP 7 — PROPOSE FINAL DATASET DESIGN

                TRAIN
                  |
       ------------------------
       |                      |
 laboratory            field images
 (PlantVillage)        (Paddy Doctor / PlantDoc)
       |                      |
       ------------------------
                  |
        duplicate-safe grouping
                  |
          TRAIN / VALIDATION

                TEST
                  |
        completely untouched
                  |
        ----------------
        |              |
     lab test       field test

*The FIELD TEST SET must NEVER be used for training, hyperparameter tuning, or model selection. It remains physically segregated.*

## STEP 8 — LEAKAGE PREVENTION

To prevent train/test contamination:
1. **Hash-based Grouping:** Compute `imagehash.phash` for all field images.
2. **Cross-Domain Leakage Check:** Compare hashes of the new field images against the existing `baseline_metrics.json`/`test_split.json` from Phase 1. Any field image that matches a lab image (same photograph used in multiple datasets) will be purged.
3. **Strict Splitting:** Near-duplicate field images (distance <= 5) will be grouped into identical splits (e.g., all go to train, or all go to test).

## STEP 9 — FINAL RECOMMENDATION

### FINAL TELANGANA CROP SCOPE
1. Rice
2. Corn/Maize
3. Tomato
4. Cotton

### FINAL DISEASE CLASSES
- Rice: Blast, Brown Spot, Bacterial Leaf Blight, Healthy.
- Corn: Blight, Rust, Gray Leaf Spot, Healthy.
- Tomato: Early Blight, Late Blight, Leaf Mold, Bacterial Spot, Healthy.
- Cotton: Bacterial Blight, Curl Virus, Healthy (requires adding to registry).

### DATASETS TO USE
- **Paddy Doctor**: Rice, 4 classes, ~5000 images, Field, Open Research. Purpose: Field Adaptation & Testing.
- **PlantDoc**: Corn/Tomato, ~1500 images, Field, CC BY 4.0. Purpose: Field Adaptation & Testing.
- **Cotton Disease (Kaggle)**: Cotton, ~2000 images, Field, CC0. Purpose: Expanding crop scope.

### DATASETS TO REJECT
- **All Kaggle Chilli & Groundnut Datasets**: Rejected due to unknown provenance, high risk of synthetic/fake data, and insufficient volume for rigorous field testing.

### DATA GAP
We lack sufficient field data for:
- Rice Sheath Blight
- Rice Leaf Scald
- Tomato Spider Mites
*These classes will rely entirely on laboratory data unless real field data is procured.*

### RECOMMENDED NEXT ML EXPERIMENT
1. Download Paddy Doctor, PlantDoc, and Cotton datasets locally.
2. Expand `class_registry.py` to support Cotton (Num Classes: 27).
3. Run `evaluate_domain_gap.py` to test the frozen baseline exclusively on the new field data, measuring the exact baseline degradation.

---

# FINAL DECISION

**TELANGANA DATA READINESS:**
PARTIALLY READY (Awaiting physical download of datasets)

**MODEL TRAINING:**
DO NOT START YET

**RECOMMENDED NEXT STEP:**
Download Paddy Doctor, PlantDoc, and Cotton datasets into `backend/data/raw_field_data`.
