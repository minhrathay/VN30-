@echo off
title VN30 Dashboard Launcher
echo ================================================
echo    VN30 FORECASTING DASHBOARD
echo ================================================
echo.
echo Starting dashboard... Please wait.
echo.

cd /d "%~dp0"

:: Activate virtual environment if exists
if exist "vn30_env\Scripts\activate.bat" (
    call vn30_env\Scripts\activate.bat
)

:: Open browser automatically
start http://localhost:8501

:: Run Streamlit
streamlit run app.py

pause
