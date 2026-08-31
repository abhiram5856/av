# NOVA API Documentation

All production API routes are versioned and exposed under `/api/v1/`. Standard liveness/version check endpoints are served directly at the root.

---

## 🛠️ Root Probes

### Health Check
* **Endpoint**: `GET /health`
* **Response**:
  ```json
  {"status": "healthy", "service": "NOVA Vision Platform"}
  ```

### Readiness Check
* **Endpoint**: `GET /ready`
* **Response**:
  ```json
  {"status": "ready"}
  ```

### Version Details
* **Endpoint**: `GET /version`
* **Response**:
  ```json
  {
    "application_version": "1.0.0",
    "model_version": "1.0.0",
    "dataset_version": "PlantVillage-Preprocessed-v1.0",
    "schema_version": "1.0.0"
  }
  ```

---

## 🌾 Version 1.0 Production Routes

### 1. Leaf Disease Diagnosis
* **Endpoint**: `POST /api/v1/diagnose/`
* **Content-Type**: `multipart/form-data`
* **Request Parameters**:
  * `image`: UploadFile (binary leaf profile)
  * `temperature`: float (default 25.0)
  * `humidity`: float (default 60.0)
  * `ph_level`: float (default 6.5)
  * `latitude`: float (default 17.3850)
  * `longitude`: float (default 78.4867)
  * `user_id`: str (default "usr_farmer")
* **Response**:
  ```json
  {
    "status": "success",
    "request_id": "req_uuid...",
    "context_hash": "3bb628...",
    "prediction": {
      "disease": "tomato_late_blight",
      "confidence": 94.5,
      "tta_enabled": true
    },
    "severity_assessment": {
      "base_vision_score": 40.0,
      "environmental_risk_factor": 1.2,
      "soil_stress_factor": 1.0,
      "final_severity_score": 48.0,
      "urgency": "High"
    },
    "root_cause_placeholder": {
      "status": "INTERFACE_PLACEHOLDER"
    },
    "gradcam_heatmap_b64": "data:image/jpeg;base64,...",
    "ai_context": { ... }
  }
  ```

### 2. PDF Report Generation
* **Endpoint**: `POST /api/v1/report`
* **Content-Type**: `application/json`
* **Request Body**:
  * `image_b64`: str (base64 encoded original leaf)
  * `heatmap_b64`: str (base64 encoded GradCAM image)
  * `disease`: str
  * `confidence`: float
  * `severity_score`: float
  * `urgency`: str
  * `weather_summary`: str
  * `rag_recommendations`: str
  * `context_hash`: str
  * `model_version`: str (optional)
  * `dataset_version`: str (optional)
  * `user_id`: str (optional)
* **Response**:
  ```json
  {
    "status": "success",
    "report_id": "uuid-report-...",
    "download_url": "/api/v1/report/uuid-report-..."
  }
  ```

### 3. Download PDF Report File
* **Endpoint**: `GET /api/v1/report/{report_id}`
* **Response**: Returns standard `application/pdf` binary stream for file saving.

### 4. Interactive Chat (RAG)
* **Endpoint**: `POST /api/v1/chat`
* **Request Body**:
  ```json
  {"query": "How do I treat early blight?"}
  ```
* **Response**:
  ```json
  {
    "response": "RAG-derived treatment suggestions from Llama3...",
    "sources": ["ICAR Guidelines Page 12"]
  }
  ```
