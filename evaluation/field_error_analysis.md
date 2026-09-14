# Field Image Diagnosis Error Analysis

## Baseline Evaluation Summary
- **Field Single-Image Accuracy:** 74.40%
- **Field Top-3 Accuracy:** 96.36%
- **Field Macro F1:** 46.60%

## Primary Root Causes of Field Failure
1. **Background Complexity:** Laboratory images feature synthetic white/black backgrounds, whereas field images contain complex soil, weed, and ambient foliage backgrounds.
2. **Class Imbalance:** Field dataset split is heavily concentrated on Rice and Cotton diseases. Unrepresented classes lower unweighted Macro F1.
3. **Lighting & Shadows:** Direct sunlight glare and shadows produce high-frequency visual noise that shifts feature representation.
4. **Visually Similar Symptoms:** Bacterial blight vs brown spot on rice leaves present similar visual necrotic spots under uncalibrated mobile camera lenses.

## Recommendations
- Retain single-source-of-truth 34-class registry (`class_registry.py`).
- Use class-weighted loss and domain-aware augmentation (crop, color jitter, blur) to generalize feature extraction across background variations.
