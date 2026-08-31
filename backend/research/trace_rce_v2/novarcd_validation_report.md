# NOVA Root Cause Dataset (NOVA-RCD) — Validation Report

**Verification Date**: July 26, 2026  
**Dataset Version**: PlantVillage-Preprocessed-v1.0-RCD  
**Total Records**: 5667  

---

## 1. Summary Metrics

| Validation Check | Status | Value / Result | Description |
| :--- | :---: | :---: | :--- |
| **Total Samples** | PASSED | 5667 | Total images in unified dataset |
| **Missing Values** | PASSED | 0 | Records with missing context keys |
| **Invalid Weather Ranges** | PASSED | 0 | Out-of-bounds weather values |
| **Impossible Combinations** | PASSED | 0 | Biologically conflicting variables |
| **Duplicates** | PASSED | 0 | Duplicate image paths |
| **Train/Val Split size** | PASSED | 5109 | Available train/val training pool |
| **Test Split size** | PASSED | 558 | Reserved test split pool |
| **Split Leakage (Overlap)** | PASSED | 0 | Mutual exclusivity check |

---

## 2. Diagnostics Category Breakdown

| Diagnostic Category | Count | Proportion | Environmental Conduciveness Profile |
| :--- | :---: | :---: | :--- |
| **HEALTHY** | 1522 | 26.86% | Dry/Moderate, normal state, severity = 0.0 |
| **FUNGAL_COOL_WET** | 756 | 13.34% | Humidity >80%, Temp <22C, Rain >40mm |
| **FUNGAL_WARM_HUMID** | 2189 | 38.63% | Humidity >80%, Temp >24C, Wind <8km/h |
| **BACTERIAL** | 600 | 10.59% | Humidity >80%, Wind >18km/h, Temp >28C |
| **VIRAL** | 400 | 7.06% | Humidity 40-60%, Temp >26C, high vectors |
| **PEST** | 200 | 3.53% | Humidity <40%, Temp >30C, rain ~0mm |

---

## 3. Split Exclusivity Check
- Checked all **558** images mapped to the Test Split against the test split registry file `test_split.json`.
- All reserved test images are strictly excluded from the train/val pool.
- **Leakage Status**: **PASSED (0.00% overlap)**.

## 4. Auditor Conclusion
The extended **NOVA-RCD** dataset is fully compliant with agricultural pathology rules, contains zero split leakages, has zero missing values, and is ready for training TRACE-RCE v2.
