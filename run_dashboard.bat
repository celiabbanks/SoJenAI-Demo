@echo off
echo [SoJenAI Demo] Starting Streamlit dashboard...
cd /d %~dp0

call venv\Scripts\activate

streamlit run .streamlit/dashboard.py

pause
