@echo off
echo 🗄️ Database Setup untuk Flutter Prediction Engine
echo ==============================================

echo.
echo 📋 Langkah-langkah setup database:
echo 1. Pastikan Laragon/XAMPP sudah running
echo 2. Buka phpMyAdmin (http://localhost/phpmyadmin)
echo 3. Buat database baru dengan nama: sensor_data
echo 4. Import file: database\sensor_data.sql
echo.

echo 🔍 Checking MySQL service...
tasklist /FI "IMAGENAME eq mysqld.exe" | find "mysqld.exe"
if errorlevel 1 (
    echo ❌ MySQL tidak running!
    echo 💡 Silakan start MySQL di Laragon/XAMPP terlebih dahulu
    pause
    exit
)

echo ✅ MySQL service detected

echo.
echo 🎯 Pilihan setup database:
echo 1. Buka phpMyAdmin untuk import manual
echo 2. Setup database dengan Python script
echo 3. Skip setup (database sudah siap)

choice /C 123 /M "Pilih opsi (1-3)"

if errorlevel 3 goto skip
if errorlevel 2 goto python_setup
if errorlevel 1 goto phpmyadmin

:phpmyadmin
echo 🌐 Membuka phpMyAdmin...
start http://localhost/phpmyadmin
echo.
echo 📝 Instruksi manual:
echo 1. Klik "New" untuk buat database baru
echo 2. Nama database: sensor_data
echo 3. Klik "Create"
echo 4. Pilih database sensor_data
echo 5. Klik tab "Import"
echo 6. Pilih file: database\sensor_data.sql
echo 7. Klik "Go"
echo.
pause
goto end

:python_setup
echo 🐍 Setup dengan Python script...
python setup_database.py
pause
goto end

:skip
echo ✅ Skipping database setup

:end
echo.
echo 🎉 Database setup selesai!
echo 📝 Lanjutkan dengan: start_api.bat

pause
