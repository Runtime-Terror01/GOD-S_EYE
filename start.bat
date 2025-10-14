@echo off
REM Surveillance System Startup Script for Windows

echo ===============================================
echo   Surveillance System - Quick Start
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_surveillance.txt
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Choose an option:
echo.
echo   1. Run example pipeline (process a video)
echo   2. Start dashboard backend only
echo   3. Run pipeline + start dashboard
echo   4. Install dependencies only
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto run_pipeline
if "%choice%"=="2" goto run_dashboard
if "%choice%"=="3" goto run_both
if "%choice%"=="4" goto install_deps

:run_pipeline
echo.
set /p video="Enter video path (e.g., input/video.mp4): "
if not exist "%video%" (
    echo Error: Video file not found: %video%
    pause
    exit /b 1
)

set /p use_vlm="Enable VLM analysis? (y/n): "
if /i "%use_vlm%"=="y" (
    set vlm_flag=--vlm
) else (
    set vlm_flag=
)

echo.
echo [3/3] Processing video...
python example_pipeline.py "%video%" %vlm_flag%

echo.
echo ===============================================
echo   Processing complete!
echo   Start dashboard to view results (option 2)
echo ===============================================
pause
exit /b 0

:run_dashboard
echo.
echo [3/3] Starting dashboard backend...
echo.
echo Dashboard will be available at:
echo   - API: http://localhost:8000
echo   - Docs: http://localhost:8000/docs
echo   - Frontend: Open dashboard\frontend.html in browser
echo.
echo Press Ctrl+C to stop the server
echo.
python dashboard/backend.py
pause
exit /b 0

:run_both
echo.
set /p video="Enter video path (e.g., input/video.mp4): "
if not exist "%video%" (
    echo Error: Video file not found: %video%
    pause
    exit /b 1
)

echo.
echo [3/3] Processing video...
python example_pipeline.py "%video%"

echo.
echo Starting dashboard in 3 seconds...
timeout /t 3 >nul

echo.
echo Dashboard will be available at:
echo   - API: http://localhost:8000
echo   - Frontend: Open dashboard\frontend.html in browser
echo.
echo Press Ctrl+C to stop the server
echo.
python dashboard/backend.py
pause
exit /b 0

:install_deps
echo.
echo [3/3] Installing dependencies...
pip install -r requirements_surveillance.txt
echo.
echo Installation complete!
pause
exit /b 0
