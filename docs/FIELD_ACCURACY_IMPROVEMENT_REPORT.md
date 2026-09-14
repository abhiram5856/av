# Field Image Accuracy Improvement Report
**AgriVision-AI Project — Final Forensic & Adaptation Analysis**

## Executive Summary
- **Baseline Architecture:** MobileNetV3-Small (34-class single source of truth registry)
- **Production Checkpoint:** `nova_mobilenet_v3_34_classes.pth` (~6.05 MB)
- **Baseline Lab Test Accuracy:** 81.88% (Macro F1: 86.57%)
- **Baseline Field Test Accuracy:** 74.40% (Macro F1: 46.60%)
- **Selected Best Candidate Model:** `BASELINE`
- **Final Field Test Accuracy:** 74.40% (Macro F1: 46.60%)
- **Field Accuracy Change:** +0.00 percentage points
- **Field Macro F1 Change:** +0.00 percentage points
- **Lab Accuracy Change:** +0.00 percentage points
- **Model Promotion Verdict:** `NO - RELEASING CURRENT STABLE PRODUCTION MODEL`

## 1. Forensic Dataset & Domain Analysis
The 34-class model was evaluated across 5,373 genuine field images partitioned into 4,812 `trainval` samples (80/20 train/val split) and 849 held-out test samples.
- **Laboratory Data Characteristics:** PlantVillage-style segmented background, studio lighting, single centered leaf.
- **Field Data Characteristics:** Real-world unsegmented crop leaves, background soil, weeds, ambient sunlight/shadows, smartphone camera blur.

## 2. Progressive Augmentation & Domain Adaptation Results
| Model / Experiment | Field Val Acc | Field Val Macro F1 | Lab Test Acc | Lab Macro F1 | Status |
|--------------------|---------------|-------------------|--------------|--------------|--------|
| **Baseline Production** | 77.26% | 55.11% | 81.88% | 86.57% | Active Baseline |
| **Exp A (Standard Preproc)** | 94.34% | 93.88% | 50.04% | 13.64% | Evaluated |
| **Exp B (+Crops & Flips)** | 95.97% | 95.20% | 52.33% | 15.72% | Evaluated |
| **Exp C (+ColorJitter)** | 94.56% | 94.16% | 51.72% | 17.12% | Evaluated |
| **Exp D (+Blur & Erasing)** | 93.36% | 92.82% | 50.84% | 13.25% | Evaluated |
| **Exp E (Field Fine-Tuning)** | 94.99% | 94.93% | 54.79% | 25.39% | Evaluated |

## 3. Final Held-Out Evaluation & Verdict
- **Field Test Accuracy:** 74.40%
- **Field Test Macro F1:** 46.60%
- **Promotion Decision:** `Candidate baseline did not meet strict promotion thresholds (Field Macro F1 change: +0.00 pp, Lab Acc: 81.88%).`

Report generated automatically at 2026-09-06 20:23:02.
