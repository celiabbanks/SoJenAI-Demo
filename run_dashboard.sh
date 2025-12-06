#!/bin/bash
echo "[SoJenAI Demo] Starting Streamlit dashboard..."
cd "$(dirname "$0")"

source venv/bin/activate

streamlit run .streamlit/dashboard.py
