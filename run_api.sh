#!/bin/bash
echo "[SoJenAI Demo] Starting FastAPI backend..."
cd "$(dirname "$0")"

source venv/bin/activate

python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
