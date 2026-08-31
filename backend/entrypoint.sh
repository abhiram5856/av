#!/bin/bash
set -e

echo "Running Alembic Database Migrations..."
alembic upgrade head

echo "Starting Uvicorn Server..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
