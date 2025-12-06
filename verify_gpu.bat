@echo off
REM ============================================
REM SoJenAI-Demo: Verify GPU availability in 'sojenai' env
REM ============================================

cd /d "%~dp0"

echo.
echo [SoJenAI-Demo] Activating conda env 'sojenai' and checking torch.cuda...
echo.

REM ---- Activate conda environment ----
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" sojenai

python - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA device detected. Check drivers and PyTorch build.")
EOF

echo.
echo [SoJenAI-Demo] GPU check complete. Press any key to close this window.
pause >nul
