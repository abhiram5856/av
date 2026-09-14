# Rejected Candidate: candidate_v3

## Status
**REJECTED**

## Reason for Rejection
The candidate model failed the reproducibility and baseline regression checks when evaluated on the authoritative inference pipeline.

During the candidate's fine-tuning phase, the data preprocessing script incorrectly used aspect-ratio-distorting image resizing (Resize((224, 224))) instead of the authoritative crop strategy. As a result, the candidate learned to expect squashed morphology and severely underperformed (-0.76% Field Macro F1, -1.13% Field Accuracy) when tested against the true production pipeline.

## Authoritative Preprocessing
Any future training, validation, or inference MUST strictly use the following preprocessing chain:
1. Resize(256)
2. CenterCrop(224) 
3. ToTensor()
4. Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

## Authoritative Production Metrics
These are the officially validated metrics for the current 34-class production checkpoint (
ova_mobilenet_v3_34_classes.pth):
- **Field Accuracy**: 80.80%
- **Field Macro F1**: 67.42%
- **Field Weighted F1**: 80.79%
- **Field Top-3**: 98.12%
- **Lab Accuracy**: 85.93%
- **Lab Macro F1**: 88.74%
- **Latency**: ~11.45 ms
- **Model Size**: 6.05 MB
