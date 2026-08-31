# GRADCAM BASELINE AUDIT

This document verifies whether the baseline MobileNetV3-Small attends to actual disease lesions (causally relevant regions) or relies on background artifacts.

## Category: Correct High

### Sample 1
- **True Class:** corn_gray_leaf_spot
- **Predicted Class:** corn_gray_leaf_spot
- **Confidence:** 0.8113
![GradCAM](images/gradcam/correct_high_0.jpg)

### Sample 2
- **True Class:** corn_leaf_blight
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.8026
![GradCAM](images/gradcam/correct_high_1.jpg)

### Sample 3
- **True Class:** corn_leaf_blight
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.9460
![GradCAM](images/gradcam/correct_high_2.jpg)

### Sample 4
- **True Class:** corn_leaf_blight
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.8763
![GradCAM](images/gradcam/correct_high_3.jpg)

## Category: Correct Low

### Sample 1
- **True Class:** corn_gray_leaf_spot
- **Predicted Class:** corn_gray_leaf_spot
- **Confidence:** 0.3068
![GradCAM](images/gradcam/correct_low_0.jpg)

### Sample 2
- **True Class:** corn_gray_leaf_spot
- **Predicted Class:** corn_gray_leaf_spot
- **Confidence:** 0.4942
![GradCAM](images/gradcam/correct_low_1.jpg)

### Sample 3
- **True Class:** corn_leaf_blight
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.4755
![GradCAM](images/gradcam/correct_low_2.jpg)

### Sample 4
- **True Class:** corn_leaf_blight
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.4158
![GradCAM](images/gradcam/correct_low_3.jpg)

## Category: Incorrect High

### Sample 1
- **True Class:** potato_late_blight
- **Predicted Class:** tomato_late_blight
- **Confidence:** 0.8764
![GradCAM](images/gradcam/incorrect_high_0.jpg)

### Sample 2
- **True Class:** rice_leaf_blast
- **Predicted Class:** rice_bacterial_leaf_blight
- **Confidence:** 0.9001
![GradCAM](images/gradcam/incorrect_high_1.jpg)

### Sample 3
- **True Class:** tomato_spider_mites_two_spotted_spider_mite
- **Predicted Class:** tomato_target_spot
- **Confidence:** 0.8339
![GradCAM](images/gradcam/incorrect_high_2.jpg)

## Category: Incorrect Low

### Sample 1
- **True Class:** corn_gray_leaf_spot
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.3303
![GradCAM](images/gradcam/incorrect_low_0.jpg)

### Sample 2
- **True Class:** corn_gray_leaf_spot
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.3876
![GradCAM](images/gradcam/incorrect_low_1.jpg)

### Sample 3
- **True Class:** corn_gray_leaf_spot
- **Predicted Class:** corn_leaf_blight
- **Confidence:** 0.4776
![GradCAM](images/gradcam/incorrect_low_2.jpg)

### Sample 4
- **True Class:** corn_leaf_blight
- **Predicted Class:** corn_gray_leaf_spot
- **Confidence:** 0.4443
![GradCAM](images/gradcam/incorrect_low_3.jpg)

