@echo off
echo 🎯 Flutter Prediction Engine - Quick Start
echo ==========================================

echo.
echo 🚀 Starting system components...

:: Start backend API
start "Backend API" cmd /k "scripts\start_api.bat"
timeout /t 3 /nobreak >nul

:: Start data simulator
start "Data Simulator" cmd /k "scripts\start_simulator.bat"
timeout /t 3 /nobreak >nul

:: Start Flutter app
echo 📱 Starting Flutter app...
call scripts\start_flutter.bat

echo.
echo 🎉 System ready!
pause
