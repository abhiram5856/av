# Knowledge Retention Fine-Tuning Report

## Final Decision: KEEP CURRENT PRODUCTION

## 1. Production Metrics (nova_mobilenet_v3_34_classes.pth)
- Field Acc: 0.8231 | Field Macro F1 (34): 0.1212 | Field Weighted F1: 0.8224
- Lab Acc: 0.8804 | Lab Macro F1: 0.9075 | Lab Weighted F1: 0.8805

## 2. New Distilled Candidate Metrics (Epoch 7)
- Field Acc: 0.7817 | Field Macro F1 (34): 0.1154 | Field Weighted F1: 0.7795
- Lab Acc: 0.8531 | Lab Macro F1: 0.8841 | Lab Weighted F1: 0.8551

## 3. Deltas
- **Field Improvement**: Acc: -0.0414 | Macro F1: -0.0058
- **Lab Regression**: Acc: -0.0273 | Macro F1: -0.0234

## 4. Training Configuration
- **Loss**: KL Divergence (T=3.0, alpha=0.5) + CrossEntropy
- **Optimizer**: AdamW (lr=5e-6)
- **Batch**: 8 x 4 (Effective 32)
- **Data Sampler**: WeightedRandomSampler (Explicit 2x oversampling of lab-only classes)
- **Backbone**: Frozen up to block 10 (highly conservative)

## 5. Class-by-Class Analysis (Lab Test)
| Class | Prod F1 | Cand F1 | Delta | Prod Recall | Cand Recall | Delta |
|---|---|---|---|---|---|---|
| chilli_healthy   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| chilli_leaf_curl [-] | 0.9677 | 0.9091 | -0.0587 | 1.0000 | 1.0000 | 0.0000 |
| chilli_leaf_spot [-] | 0.8889 | 0.7500 | -0.1389 | 0.8000 | 0.6000 | -0.2000 |
| corn_gray_leaf_spot [+] | 0.4000 | 0.5000 | 0.1000 | 0.4286 | 0.5714 | 0.1429 |
| corn_leaf_blight [-] | 0.6667 | 0.5625 | -0.1042 | 0.5789 | 0.4737 | -0.1053 |
| corn_rust_leaf [-] | 0.8000 | 0.7500 | -0.0500 | 0.8333 | 0.7500 | -0.0833 |
| cotton_bacterial_blight   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| cotton_healthy   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| cotton_leaf_curl_virus   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| groundnut_early_leaf_spot   | 1.0000 | 0.9655 | -0.0345 | 1.0000 | 0.9333 | -0.0667 |
| groundnut_healthy   | 1.0000 | 0.9677 | -0.0323 | 1.0000 | 1.0000 | 0.0000 |
| groundnut_late_leaf_spot   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| groundnut_rust   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| pepper_bell_bacterial_spot   | 0.9756 | 0.9756 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| pepper_bell_healthy   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| potato_early_blight   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| potato_healthy   | 0.9756 | 0.9302 | -0.0454 | 1.0000 | 1.0000 | 0.0000 |
| potato_late_blight   | 1.0000 | 0.9744 | -0.0256 | 1.0000 | 0.9500 | -0.0500 |
| rice_bacterial_leaf_blight [-] | 0.6917 | 0.5965 | -0.0952 | 0.6765 | 0.7500 | 0.0735 |
| rice_brown_spot   | 0.8205 | 0.7748 | -0.0457 | 0.8276 | 0.7414 | -0.0862 |
| rice_healthy   | 0.8794 | 0.8738 | -0.0056 | 0.8367 | 0.9184 | 0.0816 |
| rice_leaf_blast   | 0.8235 | 0.7988 | -0.0247 | 0.8660 | 0.7062 | -0.1598 |
| rice_leaf_scald [-] | 0.7143 | 0.6364 | -0.0779 | 0.7500 | 0.7000 | -0.0500 |
| rice_sheath_blight   | 0.8293 | 0.8500 | 0.0207 | 0.8500 | 0.8500 | 0.0000 |
| tomato_bacterial_spot   | 0.9744 | 0.9268 | -0.0475 | 0.9500 | 0.9500 | 0.0000 |
| tomato_early_blight   | 0.8421 | 0.8649 | 0.0228 | 0.8000 | 0.8000 | 0.0000 |
| tomato_healthy   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| tomato_late_blight   | 0.8293 | 0.8095 | -0.0197 | 0.8500 | 0.8500 | 0.0000 |
| tomato_leaf_mold   | 0.9756 | 0.9756 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| tomato_mosaic_virus   | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| tomato_septoria_leaf_spot [-] | 0.9231 | 0.8718 | -0.0513 | 0.9000 | 0.8500 | -0.0500 |
| tomato_spider_mites_two_spotted_spider_mite   | 0.9756 | 0.9500 | -0.0256 | 1.0000 | 0.9500 | -0.0500 |
| tomato_target_spot [-] | 0.9268 | 0.8718 | -0.0550 | 0.9500 | 0.8500 | -0.1000 |
| tomato_yellow_leaf_curl_virus   | 0.9744 | 0.9744 | 0.0000 | 0.9500 | 0.9500 | 0.0000 |


## 6. Performance & Safety
- **Test Leakage**: PASS (Strict isolation of Test sets)
- **Latency**: Candidate 119.85 ms/image
