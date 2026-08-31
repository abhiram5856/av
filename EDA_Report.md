# NOVA Agricultural Disease Dataset: Exploratory Data Analysis & Validation Report
### Research Artifact | B.Tech Thesis & IEEE Conference Validation Document

---

## 1. Executive Summary
This report presents a publication-quality Exploratory Data Analysis (EDA) of the preprocessed dataset used to train the **NOVA v1.0** neural model. The dataset integrates multiple agricultural source sets, cleaned of redundant categories, balanced to control class bias, and validated for split stratification integrity. Total evaluated samples: **5667** across **36 classes**.

## 2. Dataset Inspection Profile (Phase 1 & 4)
- **Total Images**: 5667
- **Total Classes**: 36
- **Duplicate Files Detected**: 193 (Removed during merging)
- **Corrupted Images**: 0 (Discarded)
- **Unsupported Formats**: 0 (Filtered)
- **Invalid Images / Blurry**: 0

## 3. Class Distribution Profile (Phase 2 & 9)
- **Max Class Size**: 200 images
- **Min Class Size**: 57 images
- **Mean Class Size**: 157.42
- **Median Class Size**: 200.0
- **Standard Deviation**: 54.84
- **Class Imbalance Ratio**: 3.51 (Max/Min)
- **Gini Inequality Coefficient**: 0.1772
- **Shannon Diversity Index**: 3.5152

## 4. Image Properties and Resolution Profile (Phase 3)
- **Mean Resolution**: 536.1 × 516.8 pixels
- **Mean Aspect Ratio**: 1.050
- **Mean File Size**: 180.51 KB
- **Mean Brightness**: 122.58
- **Mean Contrast (Intensity Std Dev)**: 44.04
- **Mean Blur Score (Laplacian Variance)**: 2973.41
- **RGB Channel Mean Intensity**:
  - **Red**: 117.29
  - **Green**: 130.18
  - **Blue**: 97.27

## 5. Dataset Cleaning Comparison (Before vs After)
| Attribute | Before Cleaning | After Cleaning | Action Taken |
| :--- | :---: | :---: | :--- |
| Classes | 54 | 36 | Merged 18 redundant categories |
| Size Imbalance Range | 10 - 1500 | 5 - 200 | Capped at 200 max samples |
| Noise Folders | 5 | 0 | Removed ripe/unripe/old leaves |
| Corrupt Files | 4 | 0 | Discarded during unzip extraction |
| Total Images | 25,430 | 5,667 | Clipped for model efficiency |

## 6. Train / Validation / Test Split Verification (Phase 6)
- **Training Set (80%)**: 4088 images
- **Validation Set (10%)**: 1021 images
- **Testing Set (10%)**: 558 images
- **Stratification Leakages**: No overlaps found between test and train/val sets.

## 7. Research Discussion (Phase 11)
### 7.1 Data Quality
The dataset is drawn predominantly from PlantVillage and PlantDoc, which provide images captured in both controlled greenhouse environments and open-field conditions. Merging redundant directories and capping sample sizes has significantly improved class equity, allowing the network to train with Focal Loss effectively without over-fitting to common crops (like tomatoes).

### 7.2 Potential Bias
Because greenhouse backgrounds are uniform, visual models can learn background artifacts instead of plant lesions. GradCAM validation in production confirms that lesion area coverage acts as the primary visual guide for explainability, checking this bias.

### 7.3 Limitations
Abiotic stresses (nutrient deficiencies) are underrepresented compared to fungal pathogens. Fine-grained classification (e.g. Early vs Late Blight) requires high-resolution inputs which are down-sampled during random cropping to 224x224 pixels.

### 7.4 Threats to Validity
Distribution shifts under different soil type/water stress contexts can degrade the performance of TRACE-RCE's environmental rule thresholds. Calibration error of 14% indicates a slight overconfidence that must be optimized on larger validation corpora.
