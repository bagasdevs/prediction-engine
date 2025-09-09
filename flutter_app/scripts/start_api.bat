@echo off
echo Starting API server...
cd /d "%~dp0\.."
python backend/flutter_api.py
