@echo off
echo 📊 Real-time Data Simulator
echo ============================

:: Navigate to backend folder
cd /d "%~dp0..\backend"

echo.
echo 📍 Current directory: %CD%
echo 🚀 Starting real-time sensor data simulation...
echo 📊 Data akan di-generate setiap 5 detik
echo 📱 Flutter app akan menampilkan data real-time
echo 🔄 Tekan Ctrl+C untuk stop
echo.

python realtime_simulator.py

pause
