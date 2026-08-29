@echo off
REM Batch file to run the gesture recognition data collection app
REM Simply double-click this file to start

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created.
    echo Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo Dependencies installed.
) else (
    REM Activate the virtual environment
    call venv\Scripts\activate.bat
)

REM Run the app
echo Starting gesture recognition app...
python src/app.py

REM Keep window open if there's an error
pause
