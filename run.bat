@echo off
cd /d "%~dp0"
echo Starting N64 DLSS Live...
venv310\Scripts\python.exe n64_dlss_live.py
if errorlevel 1 (
    echo.
    echo Something went wrong. Press any key to close.
    pause >nul
)
