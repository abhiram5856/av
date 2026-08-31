# CHANGELOG

All notable changes to the NOVA project will be documented in this file.

## [1.0.0] - 2026-07-25

### Added
- **API Versioning**: Serves production endpoints under `/api/v1/` prefix.
- **Health Probes**: Exposes root level endpoints `/health`, `/ready`, `/live`, and `/version`.
- **Backend PDF Generation**: Integrated `reportlab` to compile and register diagnostics reports.
- **Monitoring Instrumentation**: Measures and logs inference, GradCAM, and RAG retrieval latencies as structured JSON.
- **Deployment Profiles**: Added multi-stage `Dockerfiles` and Docker Compose recipes (`docker-compose.yml` and `docker-compose.gpu.yml`).
- **Validation Script**: Added `tests/validate_system.py` checking edge-case inputs.
- **Evaluation Script**: Added `tests/evaluate_model.py` calculating accuracy matrices on a fixed train/test split.
- **Metadata Registries**: Created `dataset/metadata.json` and `backend/models/weights/metadata.json`.

### Fixed
- **MobileNetV3 Weights Shape Mismatch**: Resolved shape classification dimension conflict (35 classes vs 36 folders check).
- **PatientHistoryContext Constructor Parameter**: Supplied default parameter to prevent constructor TypeError.
