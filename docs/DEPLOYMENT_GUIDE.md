# NOVA Deployment Guide

NOVA is ready for production deployments via containerization. It supports both CPU and GPU (NVIDIA CUDA accelerated) execution targets.

---

## 🐋 1. CPU-Only Deployment

To deploy the entire stack (FastAPI backend, Next.js frontend, PostgreSQL, and Redis) on a CPU-bound instance:

Execute from the root directory:
```bash
docker-compose up --build -d
```

Verify that all services are starting cleanly:
```bash
docker-compose ps
```

---

## ⚡ 2. GPU-Accelerated Deployment

To mount NVIDIA GPU devices into the backend container for accelerated neural network predictions:

### Prerequisites:
1. NVIDIA drivers installed on the host.
2. **NVIDIA Container Toolkit** installed and configured on the Docker runtime engine.

Execute:
```bash
docker-compose -f docker-compose.gpu.yml up --build -d
```

---

## 🩺 3. Health & Readiness Monitoring

The backend exposes the following probes for load balancers (e.g., NGINX, Kubernetes):
* **Liveness Probe**: `http://localhost:8000/live` (ensures application is running)
* **Readiness Probe**: `http://localhost:8000/ready` (ensures backend is ready to handle requests)
* **Deep Health Check**: `http://localhost:8000/health`
