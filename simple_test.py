#!/usr/bin/env python3
"""
Simple Test untuk ML Engine System
"""

def test_imports():
    """Test basic imports"""
    print("🔍 Testing imports...")
    
    try:
        import os
        import sys
        import time
        import subprocess
        from datetime import datetime
        print("✅ Basic imports OK")
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False
    
    try:
        import mysql.connector
        print("✅ MySQL connector OK")
    except Exception as e:
        print(f"❌ MySQL connector failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit OK")
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")
        return False
    
    return True

def test_database():
    """Test database connection"""
    print("\n🗄️ Testing database...")
    
    try:
        import mysql.connector
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        print(f"✅ Database OK ({count} records)")
        return True
        
    except Exception as e:
        print(f"❌ Database failed: {e}")
        return False

def test_files():
    """Test required files"""
    print("\n📁 Testing files...")
    
    import os
    
    files = [
        'start_system.py',
        'ml_engine.py',
        'ml_dashboard.py'
    ]
    
    all_ok = True
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            all_ok = False
    
    return all_ok

def main():
    print("🧪 SIMPLE ML ENGINE TEST")
    print("=" * 40)
    
    import_ok = test_imports()
    db_ok = test_database()
    files_ok = test_files()
    
    print("\n" + "=" * 40)
    print("📋 RESULTS:")
    print(f"Imports: {'✅' if import_ok else '❌'}")
    print(f"Database: {'✅' if db_ok else '❌'}")
    print(f"Files: {'✅' if files_ok else '❌'}")
    
    if import_ok and db_ok and files_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Try: python start_system.py")
    else:
        print("\n❌ SOME TESTS FAILED")
        if not db_ok:
            print("💡 Start Laragon MySQL")
            print("💡 Run: cd data_insert && python setup_initial_data.py")

if __name__ == "__main__":
    main()
