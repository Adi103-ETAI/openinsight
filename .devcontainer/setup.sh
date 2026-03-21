#!/bin/bash
set -e

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Starting Docker services (Qdrant + MongoDB + Redis)..."
docker compose up -d

echo "==> Waiting for services to be ready..."
sleep 8

echo "==> Verifying services..."
# Qdrant health check
curl -s http://localhost:6333/healthz && echo "Qdrant OK" || echo "Qdrant not ready yet"
# MongoDB check
python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27017').server_info(); print('MongoDB OK')" || echo "MongoDB not ready yet"

echo ""
echo "OpenMed dev environment ready."
echo "  FastAPI  → http://localhost:8000"
echo "  Qdrant   → http://localhost:6333"
echo "  MongoDB  → mongodb://localhost:27017"
echo "  Redis    → redis://localhost:6379"
echo ""
echo "Run: uvicorn src.api.main:app --reload --port 8000"
