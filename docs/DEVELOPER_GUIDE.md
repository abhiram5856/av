# NOVA Developer & Evaluation Guide

This guide is designed for developers, researchers, and agronomists modifying or testing the NOVA platform codebase.

---

## 🧪 Running Automated Tests

NOVA is packed with multiple automated test suites and validation harnesses:

### 1. PyTest Unit Tests
To run standard unit tests checking the AIContext builder, translation service, document chunkers, and neural vision explainers, execute from root:
```bash
pytest
```

### 2. Complete System Validation
Runs multi-boundary inputs (healthy, diseased, blurry, noise OOD, wrong objects, corrupted bytes, small & large files) checking endpoint stability and contract schemas:
```bash
$env:PYTHONPATH="."
python tests/validate_system.py
```
This writes the output validation report directly to:
`evaluation/system_validation_report.md`

### 3. Model Evaluation and Performance Metrics
Extracts classification indicators (Accuracy, Precision, Recall, Macro F1, Confusion Matrix plot, classification report) and profiles device latencies on a fixed held-out split of the dataset:
```bash
$env:PYTHONPATH="."
python tests/evaluate_model.py
```
This generates and saves the following outputs under `evaluation/`:
* `metrics.json`
* `classification_report.json`
* `confusion_matrix.png`
* `benchmark.md`

---

## 🔒 Root Cause Engine Interface Constraint

The **Root Cause Engine** is intentionally left as an abstract placeholder:
```python
# location: backend/services/root_cause_service.py
class RootCauseEnginePlaceholder:
    async def analyze(self, context: AIContext) -> dict:
        return {"status": "INTERFACE_PLACEHOLDER"}
```
> [!CAUTION]
> Do **not** modify this module or introduce causal/reasoning algorithms inside this codebase. It is designed to be populated as part of future independent major research contributions.
