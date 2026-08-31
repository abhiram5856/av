# Final Dataset Audit (27-Class Real Data Scope)

This document details the final dataset audit after removing all dummy data and integrating genuine Chilli and Cotton data.

## Overall Metrics
- **Total Images**: 10,206
- **Total Classes**: 27
- **Exact Duplicates Detected & Grouped**: 237

## Crop Distribution
- **Rice**: 4 Classes (3981 images)
- **Tomato**: 10 Classes (2000 images)
- **Corn**: 3 Classes (376 images)
- **Potato**: 3 Classes (600 images)
- **Pepper**: 2 Classes (400 images)
- **Cotton**: 1 Class (427 images - Healthy only, diseases triaged due to no open data)
- **Chilli**: 2 Classes (257 images - Leaf Curl & Leaf Spot)

## Splitting Strategy
A `StratifiedGroupKFold` split was performed using cryptographic MD5 grouping to ensure that exact duplicate images always stay within the exact same split, preventing test set leakage.

- **Train Split**: 8,164 images
- **Validation Split**: 1,021 images
- **Test Split**: 1,021 images (Untouched for final evaluation)

*The splits are deterministically locked in `evaluation/clean_*_split.json`.*
