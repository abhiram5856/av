# Final Authoritative Evaluation Report

## 1. Production Baseline (nova_mobilenet_v3_34_classes.pth)
- **Field Accuracy**: 0.8231
- **Field Macro F1**: 0.6870
- **Field Weighted F1**: 0.8224
- **Field Latency (TTA)**: 88.20 ms/image

- **Lab Accuracy**: 0.8804
- **Lab Macro F1**: 0.9075
- **Lab Weighted F1**: 0.8805
- **Lab Latency (TTA)**: 140.02 ms/image

## 2. Selected Candidate Checkpoint
Selected checkpoint: `candidate_epoch_10.pth` (Highest validation accuracy: 88.67%).

## 3. Field Test Metrics (Candidate)
- **Field Accuracy**: 0.8632
- **Field Macro F1**: 0.8511
- **Field Weighted F1**: 0.8574
- **Field Latency (TTA)**: 128.36 ms/image

## 4. Lab Test Metrics (Candidate)
- **Lab Accuracy**: 0.8698
- **Lab Macro F1**: 0.8707
- **Lab Weighted F1**: 0.8670
- **Lab Latency (TTA)**: 219.90 ms/image

## 5. Investigation of Epoch 9 -> Epoch 10 Macro F1 Drop
### Epoch 9 vs Epoch 10 Investigation (Validation Set)


## 6. Confusion Matrix (Candidate Field Test)
```
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
```
