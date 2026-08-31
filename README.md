# NOVA (Neural Optimized Vision Assistant) - Version 1.0.0

NOVA is an enterprise-grade, end-to-end explainable crop disease diagnosis and decision support platform. Built with a locked, high-concurrency microservice architecture, it enables farmers, agronomists, and researchers to diagnose crop leaf conditions, view PyTorch Grad-CAM explainability heatmaps, retrieve localized agricultural manuals via a FAISS vector database (RAG), and export diagnostic reports instantly.

---

## 🚀 Key Features

* **Visual Explainability (Grad-CAM)**: MobileNetV3-Small vision pipeline backed by test-time augmentation (TTA) and Grad-CAM heatmaps overlaying diagnostic visual attention.
* **Retrieval-Augmented Generation (RAG)**: Conversational assistant backed by FAISS vector index retrieval and ultra-fast Groq LPU (Llama 3) language generation (with local Ollama fallback).
* **Multimodal Concern Scorer**: Computes holistic concern scores and categorical concern levels dynamically factoring in vision confidence, environmental metrics, and growth stages.
* **Automated Production Logging**: Tracks request IDs, latency profiling, context hashes, and device utilization tags.
* **On-Demand Backend Reports**: Compiles ReportLab PDFs natively incorporating crop leaf snapshots, GradCAM visuals, weather details, and agronomist tips.
* **Docker Containerization**: Standard Docker Compose recipes supporting CPU execution and GPU (CUDA) hardware-mount scaling.

---

## 📁 Repository Directory Structure

```
├── backend/                   # FastAPI backend services
│   ├── api/                   # Version 1.0.0 API router endpoints
│   ├── core/                  # Middleware & exceptions
│   ├── models/                # PyTorch classifier & Grad-CAM layers
│   ├── retrieval/             # FAISS document search algorithms
│   ├── schemas/               # Frozen AIContext JSON contracts
│   └── services/              # AIContext Builders & Concern Scorer Engines
├── frontend/                  # Next.js Dashboard UI features
│   └── src/                   # React app page contexts
├── docs/                      # Architectural & usage documentation
├── evaluation/                # Model benchmarks & system metrics
├── dataset/                   # Dataset split configurations
└── docker-compose.yml         # Container configuration recipes
```

---

## 🛠️ Quick Start

Please reference the following guides to configure, run, and benchmark the application:

1. **[Installation Guide](file:///docs/INSTALLATION_GUIDE.md)**: Local Python, Next.js, and model weight setup.
2. **[Developer Guide](file:///docs/DEVELOPER_GUIDE.md)**: Testing harnesses and metrics evaluations.
3. **[API Documentation](file:///docs/API_DOCUMENTATION.md)**: REST endpoints reference list.
4. **[Deployment Guide](file:///docs/DEPLOYMENT_GUIDE.md)**: Docker Compose configs for CPU and GPU nodes.
5. **[Troubleshooting Guide](file:///docs/TROUBLESHOOTING_GUIDE.md)**: Common environment recoveries.
