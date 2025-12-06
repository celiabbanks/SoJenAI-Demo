@echo off
echo [SoJenAI Demo] Creating virtual environment and installing dependencies...
cd /d %~dp0

python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. Ensure Python is installed and on PATH.
    pause
    exit /b 1
)

call venv\Scripts\activate

python -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo [SoJenAI Demo] Environment setup complete.
pause
