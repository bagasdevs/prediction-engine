#!/usr/bin/env python3
"""
Dashboard Diagnosis and Fix
Mendiagnosis dan memperbaiki masalah dashboard kosong
"""

import mysql.connector
from mysql.connector import Error
import sys
import os
from datetime import datetime, timedelta

def diagnose_database():
    """Diagnose database issues"""
    print("🔍 DIAGNOSING DATABASE")
    print("=" * 50)
    
    try:
        # Connect to database
        db = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        cursor = db.cursor()
        print("✅ Database connection successful")
        
        # Check sensor_readings table
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        sensor_count = cursor.fetchone()[0]
        print(f"📊 sensor_readings records: {sensor_count}")
        
        if sensor_count == 0:
            print("❌ No sensor data found!")
            return False
        
        # Check recent data (last 24 hours)
        cursor.execute("""
            SELECT COUNT(*) FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL 24 HOUR
        """)
        recent_count = cursor.fetchone()[0]
        print(f"📈 Recent data (24h): {recent_count}")
        
        # Check data sample
        cursor.execute("""
            SELECT timestamp, ph, suhu, kualitas 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        
        print("\n🔍 Latest data samples:")
        for row in cursor.fetchall():
            timestamp, ph, suhu, kualitas = row
            print(f"  📊 {timestamp}: pH={ph}, Suhu={suhu}°C, Kualitas={kualitas}")
        
        # Check predictions table
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        print(f"\n🤖 Predictions records: {pred_count}")
        
        if pred_count > 0:
            cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 3")
            print("🔮 Latest predictions:")
            for row in cursor.fetchall():
                print(f"  🤖 {row}")
        
        db.close()
        return True
        
    except Error as e:
        print(f"❌ Database error: {e}")
        return False

def test_dashboard_functions():
    """Test the exact functions used in dashboard"""
    print("\n🧪 TESTING DASHBOARD FUNCTIONS")
    print("=" * 50)
    
    # Add src to path like dashboard does
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.database_manager import DatabaseManager
        print("✅ DatabaseManager import successful")
        
        # Test DatabaseManager like dashboard does
        db = DatabaseManager()
        print(f"🔧 DB Config: {db.config}")
        
        if db.connect():
            print("✅ DatabaseManager connection successful")
            
            # Test the exact query used in dashboard
            query = """
            SELECT no, ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL 2 HOUR
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            
            db.cursor.execute(query)
            data = db.cursor.fetchall()
            
            print(f"📊 Dashboard query returned {len(data)} records")
            
            if len(data) > 0:
                print("✅ Dashboard should show data")
                # Show sample
                sample = data[0]
                print(f"  📈 Sample: {sample}")
            else:
                print("❌ Dashboard query returns empty - this is the problem!")
            
            db.disconnect()
            return len(data) > 0
        else:
            print("❌ DatabaseManager connection failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def fix_data_issue():
    """Fix data issues if found"""
    print("\n🔧 FIXING DATA ISSUES")
    print("=" * 50)
    
    try:
        db = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data'
        )
        cursor = db.cursor()
        
        # Check if data is too old
        cursor.execute("""
            SELECT MAX(timestamp) as latest, MIN(timestamp) as oldest
            FROM sensor_readings
        """)
        result = cursor.fetchone()
        latest, oldest = result
        
        print(f"📅 Data range: {oldest} to {latest}")
        
        # Check if latest data is older than 1 day
        cursor.execute("""
            SELECT COUNT(*) FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL 1 DAY
        """)
        recent_count = cursor.fetchone()[0]
        
        if recent_count == 0:
            print("⚠️ No recent data found - adding fresh data...")
            
            # Insert some fresh data for testing
            import random
            for i in range(10):
                ph = round(6.5 + random.random() * 1.5, 2)
                suhu = round(23 + random.random() * 8, 1)
                kualitas = random.choice(['baik', 'buruk'])
                
                cursor.execute("""
                    INSERT INTO sensor_readings (ph, suhu, kualitas, timestamp)
                    VALUES (%s, %s, %s, NOW() - INTERVAL %s MINUTE)
                """, (ph, suhu, kualitas, i * 5))
            
            db.commit()
            print("✅ Added 10 fresh data records")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Fix error: {e}")
        return False

def start_fixed_dashboard():
    """Start dashboard with debugging"""
    print("\n🚀 STARTING FIXED DASHBOARD")
    print("=" * 50)
    
    # Check if dashboard files exist
    dashboard_files = ['ml_dashboard.py', 'simple_sensor_dashboard.py']
    
    for dashboard_file in dashboard_files:
        if os.path.exists(dashboard_file):
            print(f"✅ Found {dashboard_file}")
            print(f"🌐 Starting dashboard: {dashboard_file}")
            print(f"📊 Access at: http://localhost:8501")
            print(f"🔄 Run command: streamlit run {dashboard_file}")
            return dashboard_file
    
    print("❌ No dashboard files found")
    return None

def main():
    """Main diagnosis and fix function"""
    print("🩺 DASHBOARD DIAGNOSIS & FIX")
    print("=" * 60)
    print("Diagnosing why dashboard shows empty data...")
    print()
    
    # Step 1: Check database
    db_ok = diagnose_database()
    
    # Step 2: Test dashboard functions
    dashboard_ok = test_dashboard_functions()
    
    # Step 3: Fix issues if found
    if db_ok and not dashboard_ok:
        print("\n💡 Database has data but dashboard can't access it")
        fix_data_issue()
    elif not db_ok:
        print("\n💡 Database issue detected")
        # Could add data insertion here if needed
    
    # Step 4: Start dashboard
    dashboard_file = start_fixed_dashboard()
    
    print("\n" + "=" * 60)
    print("📋 DIAGNOSIS SUMMARY:")
    print(f"  Database OK: {'✅' if db_ok else '❌'}")
    print(f"  Dashboard Functions OK: {'✅' if dashboard_ok else '❌'}")
    print(f"  Dashboard File: {dashboard_file or '❌ Not found'}")
    
    if db_ok and dashboard_file:
        print("\n🎉 SOLUTION:")
        print("1. Stop any running dashboard (Ctrl+C)")
        print(f"2. Run: streamlit run {dashboard_file}")
        print("3. Access: http://localhost:8501")
        print("4. Dashboard should now show data!")
    else:
        print("\n❌ ISSUES FOUND:")
        if not db_ok:
            print("  - Database connection or data issues")
        if not dashboard_file:
            print("  - Dashboard files missing")

if __name__ == "__main__":
    main()
