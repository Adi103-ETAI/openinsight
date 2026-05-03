#!/bin/bash
set -e

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Starting Docker services (MongoDB + Redis)..."
docker compose up -d

echo "==> Waiting for services to be ready..."
sleep 8

echo "==> Verifying services..."
# MongoDB check
python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017').server_info(); print('MongoDB OK')" || echo "MongoDB not ready yet"

echo ""
echo "OpenInsight dev environment ready."
echo "  FastAPI  → http://localhost:8000"
echo "  VectorDB → configure VECTOR_URI / VECTOR_TOKEN in .env"
echo "  MongoDB  → mongodb://localhost:27017"
echo "  Redis    → redis://localhost:6379"
echo ""
echo "Run: uvicorn src.api.main:app --reload --port 8000"
