@echo off
setlocal

:: ============================================================
:: ms-neuron-pipeline — run pipeline
:: ============================================================
:: Activates the project virtual environment and runs the
:: full pipeline (Phases 0-4) via main_pipeline.py.
::
:: Usage:
::   scripts\run_pipeline.bat
:: Run from the project root.
:: ============================================================

:: Locate project root (one level up from this script)
set ROOT=%~dp0..

:: Activate virtual environment
call "%ROOT%\.venv\Scripts\activate.bat" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Could not activate .venv. Run install.bat first.
    exit /b 1
)

echo [INFO] Running pipeline...
python -m neuron_pipeline.main_pipeline
if errorlevel 1 (
    echo [ERROR] Pipeline exited with an error.
    exit /b 1
)

echo [INFO] Pipeline complete.
endlocal
