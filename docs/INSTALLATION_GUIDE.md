# NOVA Installation Guide

Follow these steps to set up the NOVA FastAPI backend and Next.js frontend on your local system.

---

## 📋 Prerequisites

Ensure you have the following system applications installed:
* **Python**: Version 3.10 or 3.11
* **Node.js**: Version 18 or above (with npm)
* **Ollama**: (Optional, for running local RAG llama3 model)

---

## 🐍 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   * **Windows Powershell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```
4. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Ensure model checkpoint `nova_mobilenet_v3.pth` is located at:
   `backend/models/weights/nova_mobilenet_v3.pth`

---

## 🖥️ 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm modules:
   ```bash
   npm install
   ```
3. Setup configuration variables in `.env.local`:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

---

## 🚀 3. Running the Services

### Start Backend FastAPI Server
Navigate to the repository root directory, activate virtual environment, and execute:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to view the interactive FastAPI Swagger page.

### Start Frontend Dashboard
Navigate to the `frontend/` directory and execute:
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to access the NOVA Vision Dashboard!
