@echo off
setlocal

:: ============================================================
:: ms-neuron-pipeline -- environment setup
:: ============================================================
:: Creates a .venv, installs dependencies, and installs the
:: local packages (neuron_pipeline + vastpy) in editable mode.
::
:: Usage:
::   install.bat
:: ============================================================

set VENV_DIR=.venv
set MIN_MAJOR=3
set MIN_MINOR=10

:: ------------------------------------------------------------
:: 1. Check Python is available
:: ------------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ and try again.
    exit /b 1
)

:: ------------------------------------------------------------
:: 2. Check Python version >= 3.10
:: ------------------------------------------------------------
for /f "tokens=2 delims= " %%V in ('python --version 2^>^&1') do set PY_VER=%%V
for /f "tokens=1,2 delims=." %%A in ("%PY_VER%") do (
    set PY_MAJOR=%%A
    set PY_MINOR=%%B
)

if %PY_MAJOR% LSS %MIN_MAJOR% (
    echo [ERROR] Python %PY_VER% detected. Python 3.10 or higher is required.
    exit /b 1
)
if %PY_MAJOR% EQU %MIN_MAJOR% if %PY_MINOR% LSS %MIN_MINOR% (
    echo [ERROR] Python %PY_VER% detected. Python 3.10 or higher is required.
    exit /b 1
)

echo [OK] Python %PY_VER% detected.

:: ------------------------------------------------------------
:: 3. Create virtual environment
:: ------------------------------------------------------------
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Virtual environment already exists at %VENV_DIR% -- skipping creation.
) else (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

:: ------------------------------------------------------------
:: 4. Activate virtual environment
:: ------------------------------------------------------------
call "%VENV_DIR%\Scripts\activate.bat"

:: ------------------------------------------------------------
:: 5. Upgrade pip
:: ------------------------------------------------------------
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] pip upgrade failed -- continuing anyway.
)

:: ------------------------------------------------------------
:: 6. Install dependencies from requirements.txt
:: ------------------------------------------------------------
echo [INFO] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b 1
)

:: ------------------------------------------------------------
:: 7. Install local packages in editable mode
:: ------------------------------------------------------------
echo [INFO] Installing local packages (neuron_pipeline + vastpy) in editable mode...
pip install -e .
if errorlevel 1 (
    echo [ERROR] Editable install failed.
    exit /b 1
)

:: ------------------------------------------------------------
:: 8. Done
:: ------------------------------------------------------------
echo.
echo ============================================================
echo  Installation complete.
echo  To activate the environment in a new terminal, run:
echo.
echo    %VENV_DIR%\Scripts\activate
echo ============================================================
echo.

endlocal
