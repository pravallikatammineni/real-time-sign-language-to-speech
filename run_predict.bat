@echo off
REM Batch file to run live gesture prediction
REM Simply double-click this file to start

setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Activate the virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found. Run run_app.bat first.
    pause
    exit /b 1
)

REM Check if model exists
if not exist "model\gesture_model.pkl" (
    echo Error: Trained model not found. Run run_train.bat first.
    pause
    exit /b 1
)

REM Run the prediction script
echo Starting live gesture prediction...
echo Press Q to quit.
python src/predict.py

REM Keep window open
pause
