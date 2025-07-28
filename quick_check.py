#!/usr/bin/env python3
"""
Quick System Check untuk ML Engine
"""

import os
import sys

def quick_check():
    """Quick check untuk komponen sistem"""
    print("🔍 QUICK SYSTEM CHECK")
    print("=" * 40)
    
    # Check files
    required_files = [
        'start_system.py',
        'ml_engine.py', 
        'ml_dashboard.py',
        'src/database_manager.py',
        'src/ai_models.py'
    ]
    
    print("\n📁 Checking required files...")
    all_files_ok = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_files_ok = False
    
    # Check Python modules
    print("\n📦 Checking Python modules...")
    required_modules = ['numpy', 'pandas', 'mysql.connector', 'streamlit']
    modules_ok = True
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - NOT INSTALLED")
            modules_ok = False
    
    # Check database connection
    print("\n🗄️ Checking database...")
    try:
        import mysql.connector
        connection = mysql.connector.connect(
            host='localhost',
            user='root', 
            password='',
            database='sensor_data',
            port=3306
        )
        connection.close()
        print("✅ Database connection successful")
        db_ok = True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        db_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 SYSTEM STATUS:")
    print(f"Files: {'✅ OK' if all_files_ok else '❌ MISSING'}")
    print(f"Modules: {'✅ OK' if modules_ok else '❌ INSTALL NEEDED'}")
    print(f"Database: {'✅ OK' if db_ok else '❌ CHECK MYSQL'}")
    
    if all_files_ok and modules_ok and db_ok:
        print("\n🎉 SYSTEM READY!")
        print("🚀 Run: python start_system.py")
    else:
        print("\n⚠️ SYSTEM NOT READY")
        if not modules_ok:
            print("💡 Run: pip install -r requirements.txt")
        if not db_ok:
            print("💡 Start Laragon MySQL service")
            print("💡 Run: cd data_insert && python setup_initial_data.py")

if __name__ == "__main__":
    quick_check()
