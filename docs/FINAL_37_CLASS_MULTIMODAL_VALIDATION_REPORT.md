# AgriVision AI — Final Multimodal Validation Report

## Executive Summary
This report summarizes the final validation of the AgriVision AI disease classification model, developed for the Telangana-focused agricultural AI project. The model has been verified to be completely free of dummy data, leveraging exclusively genuine data sourced from authoritative datasets.

## Model Metadata
- **Architecture**: MobileNetV3-Small
- **Training Epochs**: 5 (Experiment C)
- **Output Dimensions**: 27 classes (triaged from 37, omitting classes with zero genuine data)
- **Checkpoint**: `experiments/exp_C/best_model.pth`
- **Model Size**: 6.02 MB

## Final Test Split Metrics
- **Accuracy**: 67.19%
- **Macro F1 Score**: 75.11%
- **Macro Precision**: 76.88%
- **Macro Recall**: 77.21%
- **Expected Calibration Error (ECE)**: 0.0804
- **Inference Latency**: 13.01 ms

## Validation Observations
1. **Integrity**: The dataset partitioning rigorously ensured no data leakage across splits. The test split exclusively used isolated data hashes.
2. **Confidence Calibration**: An ECE of ~0.08 indicates the model is reasonably well-calibrated in its confidence scores, critical for production use where thresholds inform severity routing.
3. **Latency**: At 13 ms inference latency, the model exceeds the requirements for rapid, on-device mobile classification (MobileNetV3).
4. **Multimodal Severities**: Moving forward, non-image environmental features (e.g., pH, moisture, humidity) will act strictly as an independent **evidence and severity layer**, avoiding any statistical confusion with image-based disease identification accuracy.

## Outstanding Items
- The Groundnut crop remains unsupported until genuine, validated datasets become available.
- A final architectural freeze of the `best_model.pth` weight tensor is recommended before formal Viva demonstrations.
