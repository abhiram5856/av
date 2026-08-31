# CausalAg-Net: Backend Implementation Guide (With Auth & Database)

This document is a complete blueprint for the **Backend Developer**. It explains how the frontend expects the backend API to handle Authentication, Database storage, and the Core Diagnostic Engine.

---

## 1. The Global Architecture
- **Frontend (Next.js):** Handles the UI, routing (Login ➔ Dashboard), and stores Auth tokens in `localStorage`.
- **Backend (FastAPI/Node.js):** Strictly a JSON REST API. It handles user verification, database writes, and runs the AI models.
- **Database (PostgreSQL / MongoDB):** Stores `Users` and their `DiagnosticHistory`.

**CRITICAL RULE: NO SERVER-SIDE REDIRECTS!** 
Because the frontend is a React Single Page Application, your backend must **never** return a 302 Redirect (e.g., `return redirect("/dashboard")`). Your backend must return JSON tokens, and the *frontend* code will trigger the redirect in the user's browser.

---

## 2. Authentication Flow (JWT)

To secure the application, we will use JSON Web Tokens (JWT). 

### A. The `/register` and `/login` Endpoints
The frontend will send a standard JSON POST request with credentials.

**Request (Frontend ➔ Backend):**
```json
POST /login
{
  "email": "farmer@kissan.ai",
  "password": "securepassword123"
}
```

**Expected Response (Backend ➔ Frontend):**
If the credentials match your database, return a JWT token.
```json
{
  "status": "success",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "name": "Ramu"
  }
}
```
*When the frontend receives this, it will save the `access_token` and automatically redirect the user to the dashboard.*

### B. Protecting the Backend Routes
Once the user is logged in, every request the frontend makes to your backend (like the `/diagnose` endpoint) will include this HTTP header:
`Authorization: Bearer <access_token>`

Your backend must intercept requests to `/diagnose`, verify the JWT signature, and block the request (return a `401 Unauthorized`) if the token is missing or invalid.

---

## 3. Database Integration Blueprint

Your backend needs a database to store users and their past diagnoses. Here is the recommended schema:

### Table/Collection 1: `Users`
- `id` (Primary Key)
- `email` (Unique)
- `password_hash` (Bcrypt encrypted)
- `created_at`

### Table/Collection 2: `DiagnosticHistory`
- `id` (Primary Key)
- `user_id` (Foreign Key referencing Users)
- `image_url` (Where you saved the uploaded leaf image in AWS S3 or locally)
- `predicted_disease` (e.g., "Tomato_Late_Blight")
- `humidity_at_time` (Float)
- `ai_explanation` (Text)
- `timestamp`

*Workflow:* When a user hits the `/diagnose` endpoint, after the AI generates the result, your backend should `INSERT` a new row into `DiagnosticHistory` before returning the JSON to the frontend.

---

## 4. The Core Diagnostic API Contract

This is the main dashboard endpoint.

### A. The Request
- **URL:** `POST /diagnose`
- **Headers:** `Authorization: Bearer <access_token>`
- **Content-Type:** `multipart/form-data`

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `image` | `File` (Binary) | The uploaded leaf image. |
| `manual_humidity` | `String` | e.g., `"0.85"`. You must parse this to a Float. |
| `manual_leaf_wetness` | `String` | e.g., `"0.90"`. You must parse this to a Float. |

### B. The Expected Response
The frontend strictly expects this exact JSON structure to render the diagnostic graphs properly.

```json
{
  "status": "success",
  "data_source": "Manual Overrides",
  "hypothesis": "Tomato_Late_Blight",
  "neural_perception": {
    "necrotic_lesions": 0.85,
    "water_soaked_spots": 0.92,
    "chlorotic_halos": 0.12,
    "general_yellowing": 0.30,
    "healthy_tissue": 0.05
  },
  "environmental_state": {
    "high_humidity": 0.85,
    "leaf_wetness_conducive": 0.90
  },
  "logic_proof_trace": {
    "rule_evaluated": "If Tomato_Late_Blight THEN (Water Soaked Spots AND High Humidity)",
    "satisfaction_score": 0.9521,
    "is_physically_sound": true
  },
  "ai_explanation": "This appears to be Late Blight..."
}
```

---

## 5. FastAPI Implementation Example

If you are writing the backend in Python/FastAPI, here is how you handle the JWT, CORS, and Form Data all at once.

```python
from fastapi import FastAPI, Depends, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()

# 1. CORS is mandatory for Next.js to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. JWT Security Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify the JWT token here against your Database
    # If invalid, raise HTTPException(status_code=401)
    return {"user_id": 1, "email": "farmer@kissan.ai"}

# 3. The Secure Endpoint
@app.post("/diagnose")
async def run_diagnosis(
    image: UploadFile = File(...),
    manual_humidity: str = Form(...),
    manual_leaf_wetness: str = Form(...),
    current_user: dict = Depends(get_current_user) # This locks the route!
):
    
    # Process AI ...
    # Save to Database using current_user["user_id"] ...
    
    return {
        "status": "success",
        "hypothesis": "Tomato_Late_Blight",
        "neural_perception": {
            "necrotic_lesions": 0.85,
            "water_soaked_spots": 0.92,
            "healthy_tissue": 0.05
        },
        "environmental_state": {
            "high_humidity": float(manual_humidity),
            "leaf_wetness_conducive": float(manual_leaf_wetness)
        },
        "logic_proof_trace": {
            "rule_evaluated": "If Late Blight THEN Water Soaked Spots AND High Humidity",
            "satisfaction_score": 0.95,
            "is_physically_sound": True
        },
        "ai_explanation": "Replace with real LLM output."
    }
```

---

## 6. Current Frontend Prototype Status (For Backend Team)

**To the Backend/Full-Stack Teammates:**
The Next.js frontend has been fully built out as a functional prototype so you can start testing your backend APIs against a real UI immediately.

### Key Things to Know:
1. **API Utility (`src/utils/api.ts`):** 
   The frontend uses a centralized `apiFetch` wrapper. By default, it sends requests to `http://localhost:8000`. This utility automatically attaches the `Authorization: Bearer <token>` to every request!
2. **Mock Mode Enabled:**
   In `src/app/dashboard/page.tsx`, if the frontend fails to connect to your backend, it will automatically wait 1.5 seconds and display a **Mock JSON Payload**. This allows the UI transitions and styling (including a fake Grad-CAM heatmap) to be tested without needing a live backend. Once your backend is running at `localhost:8000/diagnose`, the real data will seamlessly replace the mock data.
3. **Running the Frontend:**
   Navigate into the `frontend` directory and run:
   `npm run dev`
   The UI will be available at `http://localhost:3000`.
