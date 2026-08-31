# ML Phase 2C: Domain Gap Analysis

## Overview
This report quantifies the "Domain Gap"—the drop in accuracy when our laboratory-trained baseline model is applied to real-world field data.

- **Model Checkpoint:** `nova_mobilenet_v3_26_classes_init.pth`
- **Field Test Set Size:** 849 strictly held-out images
- **Architecture:** MobileNetV3-Small (26-class initialized)

## Zero-Shot Field Metrics
| Metric | Laboratory Baseline (Phase 1) | Field Evaluation (Phase 2C Zero-Shot) | Absolute Drop |
|--------|------------------------------|--------------------------------------|---------------|
| **Accuracy** | 91.75% | 34.86% | 56.89% |
| **Macro F1** | 90.82% | 9.19% | 81.63% |

## Interpretation
As expected in real-world ML adaptation, there is a massive drop in performance when moving from clean, white-background laboratory images to noisy, varied-lighting field images. 

*(Note: The Cotton classes were randomly initialized during the surgical expansion, contributing to the drop, but even the previously learned Rice classes suffer from domain shift).*

**Conclusion:** The model is not ready for deployment. We MUST perform Domain Adaptation (Field Fine-Tuning) to recover this performance loss.
