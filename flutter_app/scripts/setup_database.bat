@echo off
echo Setting up database...
cd /d "%~dp0\.."
python backend/setup_database.py
echo Database setup completed!
pause
