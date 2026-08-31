# AgriVision AI — Multimodal Concern Score Validation Report

## Overview
This document validates the new deterministic `MultimodalConcernScorer` which replaces the legacy probabilistic severity scoring. The new engine provides a transparent 0-100 score indicating evidence concern rather than biological ground-truth severity.

## Mathematical Formulation
The concern score is calculated across 5 dimensions, capped at 100 points:
1. **Model Confidence (Max 35 points)**
   - Formula: `min(ml_confidence * 35.0, 35.0)`
   - Impact: Scales linearly with raw prediction confidence.
2. **Visual Evidence (Max 25 points)**
   - Formula: Mapped via `lesion_area_ratio` (Grad-CAM coverage).
   - Bounds: `ratio >= 0.3` = 25 pts, `ratio >= 0.1` = 15 pts, `< 0.1` = 5 pts.
3. **Environmental Compatibility (Max 20 points)**
   - Mapped dynamically: 
     - `Strongly Compatible` = 20 pts
     - `Compatible` = 15 pts
     - `Partially Compatible` = 10 pts
     - `Incompatible` / `Insufficient Evidence` = 0 pts.
4. **Growth-stage Vulnerability (Max 10 points)**
   - Seedling/Flowering: 10 pts
   - Vegetative/Fruiting: 5 pts
   - Harvest: 2 pts
5. **Image Quality (Max 10 points)**
   - Acceptable: 10 pts
   - Poor: 0 pts

## Environmental Safety Constraint
If environmental evidence is `Incompatible`, no negative penalty is applied. Instead, it contributes 0 points and explicitly flags `environmental_conflict = true`.

## Test Scenarios Passed (Identical Inputs = Identical Outputs)
1. **High model confidence + strong environmental compatibility:** Passed (Score: 98.2, Critical Attention Required).
2. **High model confidence + environmental conflict:** Passed (Score >= 75.0, Conflict Flagged).
3. **Low model confidence:** Passed (Score: 15.5, Insufficient Evidence).
4. **Missing environmental data:** Passed (Score: 65.25, High Concern).
5. **Poor image quality:** Passed (Score: 68.0, High Concern, Limiting Factors Added).
6. **Unknown growth stage:** Passed (Score: 58.0, Limiting Factors Added).
7. **Healthy prediction:** Passed (Score: 0.0, Low Concern).
8. **Strong Grad-CAM evidence:** Passed (Visual Evidence Level: Strong).
9. **Weak Grad-CAM evidence:** Passed (Visual Evidence Level: Weak).
10. **Deterministic Output:** Passed (Identical inputs yielded exact same outputs).

## Conclusion
The `MultimodalConcernScorer` acts predictably and deterministically, maintaining strict separation from probabilistic severity claims. The RAG system now respects these limitations and prompts agronomist referral when confidence drops below 60% or when environmental conditions are incompatible.
