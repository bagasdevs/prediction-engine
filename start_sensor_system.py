#!/usr/bin/env python3
"""
Real-time Data Simulator untuk Flutter App
Generate data sensor otomatis setiap 5 detik
"""

import sys
import os
import time
import subprocess
from threading import Thread

# Add project root to path
sys.path.append(os.path.dirname(__file__))

try:
    from simulasi_data.simulasi1 import SensorDataSimulator
except ImportError:
    print("⚠️ simulasi1.py not found, using fallback simulator")
    SensorDataSimulator = None

def start_continuous_simulation():
    """Start continuous data simulation"""
    print("� Starting Real-time Sensor Data Simulation")
    print("=" * 50)
    print("📊 Generating sensor data every 5 seconds")
    print("📱 Data akan tampil real-time di Flutter app")
    print("🔄 Tekan Ctrl+C untuk stop")
    print("=" * 50)
    
    if SensorDataSimulator:
        simulator = SensorDataSimulator()
        simulator.run_simulation(interval=5.0, verbose=True)
    else:
        print("❌ Simulator not available")
        
        print("✅ Simulator started successfully")
        print(f"   Process ID: {process.pid}")
        return process
    except Exception as e:
        print(f"❌ Error starting simulator: {e}")
        return None

def start_dashboard():
    """Start basic dashboard"""
    print("📊 Starting Dashboard...")
    
    try:
        # Create simple dashboard file if not exists
        dashboard_content = '''
import streamlit as st
import mysql.connector
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="🤖 Sensor Dashboard", layout="wide")

st.title("🤖 Real-time Sensor Data Dashboard")

# Database connection
@st.cache_resource
def get_db_connection():
    try:
        return mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data'
        )
    except:
        return None

# Get latest data
@st.cache_data(ttl=5)
def get_latest_data(limit=50):
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql("""
                SELECT no, ph, suhu, kualitas, timestamp 
                FROM sensor_readings 
                ORDER BY timestamp DESC 
                LIMIT %s
            """, conn, params=(limit,))
            conn.close()
            return df
        except:
            conn.close()
            return pd.DataFrame()
    return pd.DataFrame()

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Latest Sensor Readings")
    data = get_latest_data(20)
    
    if not data.empty:
        # pH Chart
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['ph'],
            mode='lines+markers',
            name='pH Level'
        ))
        fig_ph.update_layout(title="pH Levels", height=300)
        st.plotly_chart(fig_ph, use_container_width=True)
        
        # Temperature Chart
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['suhu'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='red')
        ))
        fig_temp.update_layout(title="Temperature (°C)", height=300)
        st.plotly_chart(fig_temp, use_container_width=True)
    else:
        st.warning("No data available. Make sure simulator is running.")

with col2:
    st.subheader("📈 Current Stats")
    
    if not data.empty:
        latest = data.iloc[0]
        st.metric("Latest pH", f"{latest['ph']:.2f}")
        st.metric("Latest Temperature", f"{latest['suhu']:.1f}°C")
        st.metric("Latest Quality", latest['kualitas'])
        
        # Quality distribution
        quality_counts = data['kualitas'].value_counts()
        fig_qual = go.Figure(data=[go.Pie(
            labels=quality_counts.index,
            values=quality_counts.values
        )])
        fig_qual.update_layout(title="Quality Distribution")
        st.plotly_chart(fig_qual, use_container_width=True)
        
        # Recent data table
        st.subheader("📋 Recent Data")
        st.dataframe(data.head(10), use_container_width=True)
    else:
        st.error("No data to display")

# Auto refresh
time.sleep(1)
st.rerun()
'''
        
        with open('simple_sensor_dashboard.py', 'w') as f:
            f.write(dashboard_content)
        
        # Start dashboard
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "simple_sensor_dashboard.py", "--server.port", "8501"
        ], cwd=os.getcwd())
        
        print("✅ Dashboard started at http://localhost:8501")
        return process
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def main():
    """Main function"""
    print("🚀 Starting Simplified Real-time Sensor System")
    print("=" * 50)
    
    processes = []
    
    # Start simulator
    simulator = start_mysql_simulator()
    if simulator:
        processes.append(simulator)
        time.sleep(3)  # Wait for simulator to start
    
    # Start dashboard
    dashboard = start_dashboard()
    if dashboard:
        processes.append(dashboard)
    
    if processes:
        print("\n✅ System started successfully!")
        print("📊 Dashboard: http://localhost:8501")
        print("🗄️ Simulator running in background")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)
        
        try:
            # Wait for processes
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping all services...")
            for process in processes:
                process.terminate()
            print("✅ All services stopped")
    else:
        print("❌ Failed to start system components")

if __name__ == "__main__":
    main()
