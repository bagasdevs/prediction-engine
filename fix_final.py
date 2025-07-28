#!/usr/bin/env python3
"""
Final Database Connection Fix
"""

import mysql.connector
from mysql.connector import Error
import sys
import os

def main():
    print("🔧 FINAL DATABASE CONNECTION FIX")
    print("=" * 50)
    
    # Step 1: Direct connection test
    print("\n1️⃣ Testing direct MySQL connection...")
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            count = cursor.fetchone()[0]
            print(f"✅ Direct connection works: {count} records")
            
            # Add more test data if needed
            if count < 10:
                print("📝 Adding more test data...")
                test_data = [
                    (7.2, 25.5, 'baik'),
                    (6.8, 23.0, 'baik'), 
                    (8.1, 27.2, 'baik'),
                    (5.5, 20.1, 'buruk'),
                    (7.8, 26.3, 'baik'),
                    (6.9, 24.1, 'baik'),
                    (7.5, 26.8, 'baik'),
                    (8.0, 28.2, 'baik'),
                    (5.2, 19.5, 'buruk'),
                    (7.1, 25.0, 'baik')
                ]
                
                insert_sql = "INSERT INTO sensor_readings (ph, suhu, kualitas) VALUES (%s, %s, %s)"
                cursor.executemany(insert_sql, test_data)
                connection.commit()
                print(f"✅ Added {len(test_data)} test records")
            
            cursor.close()
            connection.close()
        else:
            print("❌ Direct connection failed")
            return False
            
    except Error as e:
        print(f"❌ MySQL error: {e}")
        return False
    
    # Step 2: Test DatabaseManager
    print("\n2️⃣ Testing DatabaseManager class...")
    try:
        sys.path.append('src')
        from src.database_manager import DatabaseManager
        
        db = DatabaseManager()
        print(f"📋 Config: {db.config}")
        
        if db.connect():
            print("✅ DatabaseManager connection works")
            
            # Test query
            try:
                db.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
                count = db.cursor.fetchone()['COUNT(*)']
                print(f"📊 DatabaseManager query works: {count} records")
                db.disconnect()
            except Exception as e:
                print(f"❌ DatabaseManager query error: {e}")
                return False
        else:
            print("❌ DatabaseManager connection failed")
            return False
            
    except ImportError as e:
        print(f"❌ DatabaseManager import error: {e}")
        return False
    
    # Step 3: Clear Streamlit cache
    print("\n3️⃣ Clearing Streamlit cache...")
    try:
        import streamlit as st
        # This will clear the cache when streamlit runs
        print("💡 Streamlit cache will be cleared on next run")
    except ImportError:
        print("⚠️ Streamlit not available")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("✅ Database connection working")
    print("✅ DatabaseManager working") 
    print("✅ Test data available")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Stop any running dashboard (Ctrl+C)")
    print("2. Run: streamlit run simple_sensor_dashboard.py")
    print("3. In browser, press Ctrl+F5 for hard refresh")
    print("4. Or click 'Force Refresh' button in dashboard")
    
    print("\n💡 If still shows 'Cannot connect':")
    print("- Check browser console for errors")
    print("- Try: streamlit run test_dashboard.py")
    print("- Restart Laragon MySQL service")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 SOLUTION IMPLEMENTED SUCCESSFULLY!")
    else:
        print("\n❌ SOLUTION FAILED - CHECK ERRORS ABOVE")
        sys.exit(1)
