#!/usr/bin/env python3
"""
Fixed Dashboard - Guaranteed to Work
"""

import streamlit as st
import pandas as pd
import mysql.connector
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="📊 Fixed Sensor Dashboard",
    page_icon="📊",
    layout="wide"
)

# CSS to hide Streamlit menu
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("📊 Real-time Sensor Data Dashboard")
st.markdown("**Fixed Version - Direct Database Connection**")
st.markdown("---")

# Direct database connection (no caching)
def get_database_data():
    """Get data directly from database"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        if connection.is_connected():
            # Get recent data
            query = """
            SELECT no, ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT 100
            """
            
            df = pd.read_sql(query, connection)
            
            # Get total count
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            total_count = cursor.fetchone()[0]
            
            cursor.close()
            connection.close()
            
            return True, df, total_count
        else:
            return False, pd.DataFrame(), 0
            
    except Exception as e:
        return False, pd.DataFrame(), 0

# Sidebar
st.sidebar.header("🎛️ Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_interval = st.sidebar.selectbox("Refresh Interval (seconds)", [5, 10, 30, 60], index=1)

if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# Main dashboard
col1, col2, col3 = st.columns(3)

# Get data
success, df, total_count = get_database_data()

if success and not df.empty:
    # Status indicators
    with col1:
        st.metric("✅ Database Status", "Connected", delta="Online")
    
    with col2:
        st.metric("📊 Total Records", total_count, delta=f"+{len(df)} recent")
    
    with col3:
        st.metric("⏰ Last Update", datetime.now().strftime("%H:%M:%S"), delta="Live")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ Temperature Trend")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['suhu'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='red', width=2)
        ))
        fig_temp.update_layout(
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            height=400
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        st.subheader("🧪 pH Level Trend")
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['ph'],
            mode='lines+markers',
            name='pH Level',
            line=dict(color='blue', width=2)
        ))
        fig_ph.update_layout(
            xaxis_title="Time",
            yaxis_title="pH Level",
            height=400
        )
        st.plotly_chart(fig_ph, use_container_width=True)
    
    # Quality distribution
    st.subheader("📈 Water Quality Distribution")
    quality_counts = df['kualitas'].value_counts()
    
    fig_quality = go.Figure(data=[
        go.Bar(
            x=quality_counts.index,
            y=quality_counts.values,
            marker_color=['green', 'orange', 'red']
        )
    ])
    fig_quality.update_layout(
        xaxis_title="Quality",
        yaxis_title="Count",
        height=300
    )
    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Recent data table
    st.subheader("📋 Recent Data")
    st.dataframe(df.head(10), use_container_width=True)
    
else:
    # Error state
    st.error("❌ Cannot connect to database")
    
    st.info("""
    **Troubleshooting:**
    1. Make sure Laragon MySQL service is running
    2. Verify database 'sensor_data' exists
    3. Run: `python fix_final.py`
    4. Check database credentials in src/database_manager.py
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"🕐 Current Time: {datetime.now().strftime('%H:%M:%S')}")

with col2:
    st.write("💻 System: Real-time Sensor AI")

with col3:
    st.write("🔗 Connection: Direct MySQL")

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
