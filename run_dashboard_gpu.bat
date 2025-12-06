@echo off
REM ============================================
REM SoJenAI-Demo: Start Streamlit dashboard UI
REM ============================================

REM Change to the folder where this script lives (SoJenAI-Demo)
cd /d "%~dp0"

echo.
echo [SoJenAI-Demo] Activating conda env 'sojenai' and starting Streamlit dashboard...
echo.

REM ---- Activate conda environment ----
REM If this line fails, update the path to your Anaconda install.
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" sojenai

REM Optional: force use of GPU 0 (not strictly needed for UI, but harmless)
set CUDA_VISIBLE_DEVICES=0

REM ---- Start Streamlit UI ----
python -m streamlit run .streamlit\dashboard.py

echo.
echo [SoJenAI-Demo] Dashboard stopped. Press any key to close this window.
pause >nul
