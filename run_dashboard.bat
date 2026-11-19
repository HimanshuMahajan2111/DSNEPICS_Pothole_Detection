@echo off
title ROADAUDIT AI - Streamlit Dashboard
echo =========================================================================
echo ROADAUDIT.AI // 3D POTHOLE FUSION ENGINE
echo SKEUOBRUTALIST STREAMLIT LAUNCHER
echo =========================================================================
cd /d "%~dp0"

netstat -ano | findstr :8501 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Streamlit server is ALREADY running on http://localhost:8501!
    echo Opening browser to http://localhost:8501 ...
    start http://localhost:8501
    echo.
    echo If you want to restart the server, close the previous Python window or press Ctrl+C there.
    pause
    exit /b
)

echo Starting Streamlit Web Server on http://localhost:8501 ...
start http://localhost:8501
python -m streamlit run streamlit_app\app.py --server.port 8501
pause
