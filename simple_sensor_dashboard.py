#!/usr/bin/env python3
"""
Simple Real-time Sensor Dashboard
Monitoring sensor data tanpa AI models (untuk testing)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database_manager import DatabaseManager
except ImportError:
    st.error("❌ Cannot import DatabaseManager. Check if src/database_manager.py exists.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="📊 Sensor Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = False

# Remove caching temporarily for debugging
def get_recent_data(hours=2):
    """Get recent sensor data"""
    try:
        db = DatabaseManager()
        print(f"🔍 DB Config: {db.config}")  # Debug line
        
        if db.connect():
            print("✅ Database connected successfully")  # Debug line
            query = f"""
            SELECT no, ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL {hours} HOUR
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            
            # Execute query manually since fetch_data might have issues
            db.cursor.execute(query)
            data = db.cursor.fetchall()
            db.disconnect()
            
            if data:
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=['no', 'ph', 'suhu', 'kualitas', 'timestamp'])
                st.session_state.connection_status = True
                print(f"📊 Retrieved {len(df)} records")  # Debug line
                return df
            else:
                st.session_state.connection_status = True
                print("⚠️ No data found")  # Debug line
                return pd.DataFrame()
        else:
            st.session_state.connection_status = False
            print("❌ Database connection failed")  # Debug line
            return pd.DataFrame()
    except Exception as e:
        st.session_state.connection_status = False
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_database_stats():
    """Get basic database statistics"""
    try:
        db = DatabaseManager()
        if db.connect():
            # Get basic stats
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                AVG(ph) as avg_ph,
                AVG(suhu) as avg_suhu,
                SUM(CASE WHEN kualitas = 'baik' THEN 1 ELSE 0 END) as good_count,
                SUM(CASE WHEN kualitas = 'buruk' THEN 1 ELSE 0 END) as bad_count,
                MAX(timestamp) as latest_reading
            FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL 1 HOUR
            """
            
            db.cursor.execute(stats_query)
            result = db.cursor.fetchone()
            db.disconnect()
            
            if result:
                return {
                    'total_records': result[0],
                    'avg_ph': result[1] or 0,
                    'avg_suhu': result[2] or 0,
                    'good_count': result[3] or 0,
                    'bad_count': result[4] or 0,
                    'latest_reading': result[5]
                }
        return {}
    except Exception as e:
        st.error(f"Stats error: {str(e)}")
        return {}

def test_database_connection():
    """Test database connection"""
    try:
        db = DatabaseManager() 
        print(f"🔍 Testing connection with config: {db.config}")  # Debug
        
        if db.connect():
            print("✅ Connection successful, testing query...")  # Debug
            db.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            count = db.cursor.fetchone()[0]
            db.disconnect()
            print(f"📊 Found {count} records")  # Debug
            return True, count
        else:
            print("❌ Connection failed in test function")  # Debug
            return False, "Connection failed"
    except Exception as e:
        print(f"❌ Exception in test: {str(e)}")  # Debug
        return False, str(e)

def create_time_series_chart(df, column, title, color='blue'):
    """Create time series chart"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title=title, height=300)
        return fig
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df[column],
        mode='lines+markers',
        name=column.title(),
        line=dict(color=color, width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=column.title(),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.title("📊 Real-time Sensor Data Dashboard")
    st.markdown("**Simple monitoring dashboard for sensor data**")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    # Refresh interval
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (seconds)",
        [5, 10, 15, 30, 60],
        index=1
    )
    
    # Manual refresh
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # System status
    st.sidebar.subheader("🔧 System Status")
    
    # Test database connection
    conn_ok, db_info = test_database_connection()
    if conn_ok:
        st.sidebar.success("✅ Database: Connected")
        st.sidebar.metric("Total Records", db_info)
    else:
        st.sidebar.error("❌ Database: Disconnected")
        st.sidebar.error(f"Error: {db_info}")
    
    # Data period selector
    data_hours = st.sidebar.selectbox(
        "Data Period (hours)",
        [1, 2, 6, 12, 24],
        index=1
    )
    
    # Main content
    if conn_ok:
        # Get data
        df = get_recent_data(hours=data_hours)
        stats = get_database_stats()
        
        # Statistics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Records (1h)", stats.get('total_records', 0))
        
        with col2:
            avg_ph = stats.get('avg_ph', 0)
            st.metric("Avg pH (1h)", f"{avg_ph:.2f}")
        
        with col3:
            avg_temp = stats.get('avg_suhu', 0)
            st.metric("Avg Temp (1h)", f"{avg_temp:.1f}°C")
        
        with col4:
            total_quality = stats.get('good_count', 0) + stats.get('bad_count', 0)
            if total_quality > 0:
                good_pct = (stats.get('good_count', 0) / total_quality) * 100
                st.metric("Good Quality %", f"{good_pct:.1f}%")
            else:
                st.metric("Good Quality %", "0%")
        
        # Charts section
        if not df.empty:
            st.markdown("---")
            st.subheader(f"📈 Data Visualization (Last {data_hours} hours)")
            
            # Time series charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                ph_chart = create_time_series_chart(df, 'ph', 'pH Levels Over Time', 'blue')
                st.plotly_chart(ph_chart, use_container_width=True)
            
            with chart_col2:
                temp_chart = create_time_series_chart(df, 'suhu', 'Temperature Over Time', 'red')
                st.plotly_chart(temp_chart, use_container_width=True)
            
            # Quality analysis
            st.markdown("---")
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                # Quality distribution pie chart
                if 'kualitas' in df.columns:
                    quality_counts = df['kualitas'].value_counts()
                    
                    fig_pie = px.pie(
                        values=quality_counts.values,
                        names=quality_counts.index,
                        title=f"Quality Distribution (Last {data_hours}h)",
                        color_discrete_map={
                            'baik': 'green', 
                            'buruk': 'red'
                        }
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with quality_col2:
                # Quality over time
                df_quality_numeric = df.copy()
                quality_map = {'baik': 2, 'buruk': 1}
                df_quality_numeric['quality_score'] = df_quality_numeric['kualitas'].map(quality_map)
                
                fig_quality = go.Figure()
                fig_quality.add_trace(go.Scatter(
                    x=df_quality_numeric['timestamp'],
                    y=df_quality_numeric['quality_score'],
                    mode='markers',
                    name='Quality Score',
                    marker=dict(
                        size=8,
                        color=df_quality_numeric['quality_score'],
                        colorscale=[[0, 'red'], [0.5, 'orange'], [1, 'green']],
                        showscale=True,
                        colorbar=dict(
                            title="Quality",
                            tickvals=[1, 2],
                            ticktext=['Buruk', 'Baik']
                        )
                    )
                ))
                
                fig_quality.update_layout(
                    title=f"Quality Over Time (Last {data_hours}h)",
                    xaxis_title="Time",
                    yaxis_title="Quality Score",
                    yaxis=dict(tickvals=[1, 2], ticktext=['Buruk', 'Baik']),
                    height=400
                )
                st.plotly_chart(fig_quality, use_container_width=True)
            
            # Recent data table
            st.markdown("---")
            st.subheader("🔍 Recent Readings")
            
            # Show last 20 records
            recent_display = df.head(20).copy()
            if not recent_display.empty:
                # Format timestamp
                recent_display['timestamp'] = pd.to_datetime(recent_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display table
                st.dataframe(
                    recent_display[['timestamp', 'ph', 'suhu', 'kualitas']], 
                    use_container_width=True
                )
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("⚠️ No data available for the selected period")
            st.info("💡 Make sure the sensor simulator is running: `python simulasi.py`")
    
    else:
        st.error("❌ Cannot connect to database")
        st.info("""
        **Troubleshooting:**
        1. Make sure MySQL service is running (check Laragon)
        2. Verify database 'sensor_data' exists
        3. Check database credentials in src/database_manager.py
        4. Run `python mysql_test.py` to test connection
        """)
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**📊 Sensor Data Dashboard**")
    
    with footer_col2:
        st.markdown(f"**🕒 Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    with footer_col3:
        if st.button("🔄 Force Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # Auto refresh
    if st.session_state.auto_refresh and conn_ok:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
