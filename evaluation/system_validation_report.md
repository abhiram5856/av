# NOVA Complete System Validation Report

Executed on: 2026-07-25 17:33:26

## Scenario Metrics Summary

| Scenario | Target Status Code | Observed Status | Status Pass | Notes |
|---|---|---|---|---|
| Healthy Leaf | 200 | PASS | Checked contract fields. |
| Diseased Leaf | 200 | PASS | Checked contract fields. |
| Blurry Leaf | 200 | PASS | Checked contract fields. |
| Ood Image | 200 | PASS | Checked contract fields. |
| Wrong Object | 200 | PASS | Checked contract fields. |
| Empty Upload | 500 | PASS | Checked contract fields. |
| Corrupted File | 500 | PASS | Checked contract fields. |
| Large Image | 200 | PASS | Checked contract fields. |
| Small Image | 200 | PASS | Checked contract fields. |

## Detailed API Validation Assertions

### AIContext Verification (Healthy Leaf)

- **Correlation / Request ID**: `req_121e49af-14ca-4a7a-a62e-8107e59725a8`
- **Context Hash**: `d43b951d8c519d60b90d881bf3f7df28b3e0b955fd166f91ea31dbc7e4d883fe`
- **Predicted Disease**: `tomato_late_blight`
- **ML Confidence**: `95.05%`
- **Urgency Score**: `Low`
- **Root Cause placeholder**: `{'status': 'INTERFACE_PLACEHOLDER', 'primary_root_cause': 'Pending Algorithmic Implementation', 'confidence_score': 0.0, 'contributing_factors': []}`
- **GradCAM Heatmap Generated**: Yes (Base64)


---
*Report generated automatically by validate_system.py test harness.*