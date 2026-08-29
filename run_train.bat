@echo off
REM Batch file to train the gesture recognition model
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

REM Check if dataset exists
if not exist "data\gesture_dataset.csv" (
    echo Error: Dataset not found. Run run_app.bat first and collect gesture data.
    pause
    exit /b 1
)

REM Run the training script
echo Starting model training...
python src/train_model.py

REM Keep window open
pause
