#!/usr/bin/env python3
"""
Simple Database Connection Test
"""

import mysql.connector
from mysql.connector import Error

# Test connection
try:
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='sensor_data',
        port=3306
    )
    
    if connection.is_connected():
        print("✅ Database connection successful!")
        
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        count = cursor.fetchone()[0]
        print(f"📊 Records in database: {count}")
        
        cursor.close()
        connection.close()
        
        print("🎉 Dashboard should work now!")
        print("🚀 Start dashboard with: streamlit run simple_sensor_dashboard.py")
    else:
        print("❌ Connection failed")
        
except Error as e:
    print(f"❌ Database error: {e}")
    print("💡 Make sure Laragon MySQL is running")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
