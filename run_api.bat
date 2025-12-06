@echo off
echo [SoJenAI Demo] Starting FastAPI backend...
cd /d %~dp0

call venv\Scripts\activate

python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload

pause
