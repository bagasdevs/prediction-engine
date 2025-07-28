#!/usr/bin/env python3
"""
Simple Dashboard Test - Direct Database Connection
"""

import streamlit as st
import pandas as pd
import mysql.connector
import time
from datetime import datetime

st.set_page_config(
    page_title="🔧 Database Connection Test",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 Database Connection Test")
st.markdown("---")

# Direct database connection test
def test_direct_connection():
    """Test direct MySQL connection"""
    try:
        # Direct connection without DatabaseManager
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) as total FROM sensor_readings")
            result = cursor.fetchone()
            count = result['total']
            
            # Get sample data
            cursor.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 5")
            sample_data = cursor.fetchall()
            
            cursor.close()
            connection.close()
            
            return True, count, sample_data
        else:
            return False, "Connection failed", []
            
    except Exception as e:
        return False, str(e), []

# Test connection
st.subheader("🔍 Direct Connection Test")

if st.button("Test Connection"):
    with st.spinner("Testing database connection..."):
        success, result, sample = test_direct_connection()
        
        if success:
            st.success(f"✅ Connection successful! Found {result} records")
            
            if sample:
                st.subheader("📊 Sample Data")
                df = pd.DataFrame(sample)
                st.dataframe(df)
            else:
                st.warning("⚠️ No data found in database")
                
        else:
            st.error(f"❌ Connection failed: {result}")

# Alternative test using DatabaseManager
st.subheader("🔍 DatabaseManager Test")

if st.button("Test DatabaseManager"):
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.database_manager import DatabaseManager
        
        with st.spinner("Testing DatabaseManager..."):
            db = DatabaseManager()
            st.write(f"📋 Config: {db.config}")
            
            if db.connect():
                st.success("✅ DatabaseManager connection successful!")
                
                # Test query
                try:
                    db.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
                    count = db.cursor.fetchone()[0]
                    st.info(f"📊 Records found: {count}")
                    
                    db.disconnect()
                except Exception as e:
                    st.error(f"❌ Query error: {e}")
            else:
                st.error("❌ DatabaseManager connection failed")
                
    except ImportError as e:
        st.error(f"❌ Import error: {e}")
    except Exception as e:
        st.error(f"❌ DatabaseManager error: {e}")

# Auto refresh option
st.sidebar.header("🎛️ Controls")
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# Show current time
st.sidebar.write(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")

# Instructions
st.markdown("---")
st.markdown("""
### 💡 Instructions:
1. **Test Connection** - Direct MySQL connection test
2. **Test DatabaseManager** - Test using the DatabaseManager class
3. If both fail, check:
   - Laragon MySQL is running
   - Database 'sensor_data' exists
   - Port 3306 is accessible
""")

# Show connection details
st.markdown("---")
st.subheader("📋 Connection Details")
st.code("""
Host: localhost
Port: 3306
User: root
Password: (empty)
Database: sensor_data
""")

# Auto refresh every 10 seconds
time.sleep(0.1)
st.rerun()
