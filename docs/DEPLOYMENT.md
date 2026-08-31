# AgriVision AI - Deployment Guide

This guide outlines how to deploy the AgriVision AI Next.js frontend to Vercel and the FastAPI backend.

## 1. Frontend Deployment (Vercel)

The frontend is a Next.js application that can be seamlessly deployed on Vercel. 

### Prerequisites
- A GitHub repository containing the frontend code.
- A Vercel account linked to your GitHub.
- A Supabase project.

### Deployment Steps
1. Push your local `frontend` directory to a new GitHub repository (or the root if it's a monorepo, just set the root directory in Vercel to `frontend`).
2. Log into Vercel and click **Add New Project**.
3. Import your GitHub repository.
4. If your Next.js code is in a `frontend` folder, make sure the **Root Directory** is set to `frontend`.
5. Under **Environment Variables**, you **MUST** add the following variables:
   - `NEXT_PUBLIC_API_BASE_URL` (e.g., `https://your-backend-api.com` or keep it `http://localhost:8000` if you haven't deployed the backend yet)
   - `NEXT_PUBLIC_SUPABASE_URL` (From Supabase -> Project Settings -> API)
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` (From Supabase -> Project Settings -> API)
6. Click **Deploy**. Vercel will build and host your Next.js app.

## 2. Backend Deployment (FastAPI)

The backend is built with FastAPI and requires Python 3.10+, PyTorch, and PostgreSQL (via Supabase).

### Deployment Steps (Render, Railway, or AWS)
1. Ensure the `backend` folder contains a `requirements.txt` (or Pipfile).
2. Set the start command to: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Set your backend environment variables:
   - `DATABASE_URL` (Your Supabase PostgreSQL connection string)
   - `JWT_SECRET_KEY` (Used for Supabase Auth decoding if necessary)
4. Once deployed, note the public URL.

## 3. Post-Deployment Linking (CRITICAL)

After deploying both:
1. Copy your Vercel frontend URL.
2. In your backend `main.py`, ensure the Vercel URL is listed in the `allow_origins` array (or matches the regex). 
3. Go back to Vercel, update the `NEXT_PUBLIC_API_BASE_URL` environment variable to your deployed backend URL.
4. Redeploy the Vercel project to apply the new environment variable.

## Security Reminders
- **DO NOT** expose your Supabase `service_role` key in Vercel.
- **DO NOT** commit `.env.local` to GitHub.
- Ensure file uploads are restricted to <= 10MB in your backend configuration (this is already set in `diagnose.py`).
