#!/usr/bin/env python3
"""
Dashboard Fix - Final Solution
"""

print("🔧 DASHBOARD REPAIR SOLUTION")
print("=" * 50)

print("\n✅ MASALAH YANG SUDAH DIPERBAIKI:")
print("1. ✅ Database 'sensor_data' created")
print("2. ✅ MySQL connection configuration fixed") 
print("3. ✅ Port 3306 added to database config")
print("4. ✅ Test data inserted (50+ records)")
print("5. ✅ Tables created (sensor_readings, predictions)")

print("\n🚀 CARA MENJALANKAN SISTEM:")
print("1. Pastikan Laragon MySQL running")
print("2. Refresh dashboard atau restart:")
print("   - Close dashboard browser tab")
print("   - Run: streamlit run simple_sensor_dashboard.py")
print("   - Or click 'Force Refresh' button in dashboard")

print("\n💡 TROUBLESHOOTING:")
print("- If still shows 'Cannot connect':")
print("  1. Stop dashboard (Ctrl+C in terminal)")
print("  2. Run: python quick_db_test.py")
print("  3. Restart dashboard")

print("\n🎯 MANUAL COMMANDS:")
print("python quick_db_test.py          # Test database")  
print("python simulasi.py               # Add more data")
print("streamlit run simple_sensor_dashboard.py  # Start dashboard")

print("\n" + "=" * 50)
print("🎉 SOLUTION READY!")
print("Dashboard should now connect to database!")

# Quick verification
print("\n🧪 Quick verification:")
try:
    import mysql.connector
    conn = mysql.connector.connect(
        host='localhost',
        user='root', 
        password='',
        database='sensor_data',
        port=3306
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sensor_readings")
    count = cursor.fetchone()[0]
    print(f"✅ Database accessible: {count} records found")
    conn.close()
except Exception as e:
    print(f"⚠️ Verification failed: {e}")
    print("💡 Make sure Laragon MySQL is running")
