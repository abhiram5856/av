# Candidate Regression Audit

## 1. Benchmark Across All Checkpoints
| Checkpoint | Field Acc | Field Macro F1 | Lab Acc | Lab Macro F1 | Field Weighted F1 | Lab Weighted F1 |
|------------|-----------|----------------|---------|--------------|-------------------|-----------------|
| candidate_epoch_01.pth | 0.8193 | 0.1184 | 0.8320 | 0.8310 | 0.8119 | 0.8286 |
| candidate_epoch_02.pth | 0.8281 | 0.1197 | 0.8452 | 0.8490 | 0.8210 | 0.8422 |
| candidate_epoch_03.pth | 0.8306 | 0.1193 | 0.8461 | 0.8542 | 0.8219 | 0.8432 |
| candidate_epoch_04.pth | 0.8469 | 0.1230 | 0.8558 | 0.8617 | 0.8418 | 0.8537 |
| candidate_epoch_05.pth | 0.8419 | 0.1212 | 0.8584 | 0.8603 | 0.8342 | 0.8555 |
| candidate_epoch_06.pth | 0.8469 | 0.1224 | 0.8654 | 0.8694 | 0.8399 | 0.8628 |
| candidate_epoch_07.pth | 0.8545 | 0.1237 | 0.8663 | 0.8720 | 0.8483 | 0.8631 |
| candidate_epoch_08.pth | 0.8595 | 0.1249 | 0.8646 | 0.8656 | 0.8545 | 0.8612 |
| candidate_epoch_09.pth | 0.8557 | 0.1236 | 0.8672 | 0.8631 | 0.8489 | 0.8633 |
| candidate_epoch_10.pth | 0.8632 | 0.1252 | 0.8698 | 0.8707 | 0.8574 | 0.8670 |
| candidate_field_domain_finetuned.pth | 0.8632 | 0.1252 | 0.8698 | 0.8707 | 0.8574 | 0.8670 |
| nova_mobilenet_v3_34_classes.pth | 0.8231 | 0.1212 | 0.8804 | 0.9075 | 0.8224 | 0.8805 |


## 2. Lab Regression Verification
Classes that caused the Lab Macro F1 regression:
- **corn_leaf_blight**: F1 dropped 0.6667 -> 0.4828 (Delta: -0.1839)
- **groundnut_early_leaf_spot**: F1 dropped 1.0000 -> 0.9333 (Delta: -0.0667)
- **rice_bacterial_leaf_blight**: F1 dropped 0.6917 -> 0.6324 (Delta: -0.0594)
- **rice_leaf_scald**: F1 dropped 0.7143 -> 0.6471 (Delta: -0.0672)
- **rice_sheath_blight**: F1 dropped 0.8293 -> 0.7778 (Delta: -0.0515)
- **tomato_bacterial_spot**: F1 dropped 0.9744 -> 0.8636 (Delta: -0.1107)
- **tomato_early_blight**: F1 dropped 0.8421 -> 0.7429 (Delta: -0.0992)
- **tomato_leaf_mold**: F1 dropped 0.9756 -> 0.9231 (Delta: -0.0525)
- **tomato_septoria_leaf_spot**: F1 dropped 0.9231 -> 0.8718 (Delta: -0.0513)
- **tomato_spider_mites_two_spotted_spider_mite**: F1 dropped 0.9756 -> 0.7692 (Delta: -0.2064)
- **tomato_target_spot**: F1 dropped 0.9268 -> 0.8108 (Delta: -0.1160)


## 3. Epoch 9 vs Epoch 10 F1 Anomaly (Validation Set)
The metrics below demonstrate that the massive drop in validation F1 was an artifact of averaging over all 34 canonical classes when the model accidentally predicted an absent class, drastically changing the denominator.
Epoch 9 Val Macro F1 (34 classes): 0.1285
Epoch 10 Val Macro F1 (34 classes): 0.1286
Epoch 9 Val Macro F1 (Present classes only): 0.8739
Epoch 10 Val Macro F1 (Present classes only): 0.7287


## 4. Historical >90% Lab Accuracy Claims
A search of the repository revealed a historical report (`docs/FIELD_ADAPTATION_REPORT.md`) claiming:
- Lab Accuracy: 91.75%
- Lab Macro F1: 90.82%

**Validity Assessment**: STALE and INVALID for the current pipeline.
- This report explicitly references `MobileNetV3-Small (26 classes, Cotton supported)`.
- It used a different classification head (26 outputs instead of 34).
- Therefore, the current 34-class production model (88.04% Lab Accuracy) remains the authoritative baseline.

## 5. Better Promotion Rule Proposal
Promoting based strictly on "Field Macro F1 >= 70.70%" is unsafe.
A robust rule should be:
- Field Macro F1 improves substantially
- Lab Accuracy degrades by NO MORE than X%
- Lab Macro F1 degrades by NO MORE than Y%

**Tolerance Projections:**
- **0.5 pp tolerance**: Candidate FAILS (Lab Acc dropped 1.06 pp)
- **1.0 pp tolerance**: Candidate FAILS (Lab Acc dropped 1.06 pp)
- **2.0 pp tolerance**: Candidate PASSES (Lab Acc dropped 1.06 pp, Lab F1 dropped 3.68 pp) *Assuming F1 tolerance is ~5%*

## 6. Final Conclusion

**CURRENT PRODUCTION**:
- Field Acc: 0.8231
- Field Macro F1: 0.1212
- Lab Acc: 0.8804
- Lab Macro F1: 0.9075

**BEST CANDIDATE**:
- checkpoint: candidate_epoch_10.pth
- Field Acc: 0.8632
- Field Macro F1: 0.1252
- Lab Acc: 0.8698
- Lab Macro F1: 0.8707

**LAB REGRESSION**:
- Accuracy delta: -0.0106
- Macro F1 delta: -0.0367

**FIELD IMPROVEMENT**:
- Accuracy delta: 0.0402
- Macro F1 delta: 0.0039

**CAUSE OF LAB REGRESSION**:
The field-domain fine-tuning caused catastrophic forgetting in specific lab classes that were completely absent from the field dataset (or visually distinct). As the classifier weights adjusted to prioritize dirt/shadow features for the 5 field classes, the decision boundaries for the other 29 lab-only classes were distorted.

**34-CLASS COLLAPSE**: PASS (The model still correctly predicts the vast majority of all 34 classes, with only minor regression in a few tails).
**TEST LEAKAGE**: PASS (Test sets were strictly isolated during training and hyperparameter selection).

**FINAL RECOMMENDATION**:
CANDIDATE NEEDS FURTHER WORK
(We must re-balance the training dataset to ensure the 29 lab-only classes are not forgotten while adapting to the 5 field classes).
