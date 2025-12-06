
@echo off
REM ============================================
REM SoJenAI-Demo: Start FastAPI backend on GPU
REM ============================================

REM Change to the folder where this script lives (SoJenAI-Demo)
cd /d "%~dp0"


echo.
echo [SoJenAI-Demo] Activating conda env 'sojenai' and starting FastAPI backend...
echo.

REM ---- Activate conda environment ----
REM If this line fails, update the path to your Anaconda install.
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" sojenai

REM Optional: force use of GPU 0
set CUDA_VISIBLE_DEVICES=0

REM ---- Start the FastAPI app on 127.0.0.1:8010 ----
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload

echo.
echo [SoJenAI-Demo] Backend stopped. Press any key to close this window.
pause >nul
