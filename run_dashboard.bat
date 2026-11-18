@echo off
title ROADAUDIT AI - Streamlit Dashboard
echo =========================================================================
echo ROADAUDIT.AI // 3D POTHOLE FUSION ENGINE
echo SKEUOBRUTALIST STREAMLIT LAUNCHER
echo =========================================================================
cd /d "d:\DSN3099-EPICS\Implementation"
echo Starting Streamlit on http://localhost:8501 ...
start http://localhost:8501
python -m streamlit run streamlit_app\app.py --server.port 8501
pause
