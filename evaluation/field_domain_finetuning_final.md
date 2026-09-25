# Field-Domain Fine-Tuning Final Report

## Hardware / Settings
- Device: CUDA
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Batch Size: 8 (Physical) x 4 (Accumulation) = 32 (Effective)
- AMP Enabled: True
- Dataset Memory Strategy: Lazy Load (`Image.open` in `__getitem__` inside a context manager)

## Dataset Distribution
- Field Train: 3658
- Field Val: 918
- Field Test: 797
- Lab Train: 9101
- Lab Val/Test: 1137

## Production Baseline (5-View TTA)
- Field Accuracy: 0.8231
- Field Macro F1: 0.6870
- Lab Accuracy: 0.8804
- Lab Macro F1: 0.9075

## Candidate (5-View TTA)
- Field Accuracy: 0.8632
- Field Macro F1: 0.8511
- Lab Accuracy: 0.8698
- Lab Macro F1: 0.8707

## Decision
**SAFE TO PROMOTE CANDIDATE**