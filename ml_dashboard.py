#!/usr/bin/env python3
"""
ML Output Dashboard - Real-time Machine Learning Results
Fokus pada visualisasi output AI models dan prediksi ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import Error
import time
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="🧠 ML Output Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.prediction-card {
    background-color: #e8f4fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #00cc96;
}
.model-status {
    font-size: 0.8rem;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# Database connection function
@st.cache_resource
def get_database_connection():
    """Get database connection with caching"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        return connection
    except Error as e:
        st.error(f"Database connection error: {e}")
        return None

def get_iot_data(limit=1000):
    """Get latest IoT sensor data"""
    try:
        connection = get_database_connection()
        if connection:
            cursor = connection.cursor()
            query = """
            SELECT timestamp, ph, suhu, kualitas 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            cursor.execute(query, (limit,))
            data = cursor.fetchall()
            cursor.close()
            
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'ph', 'suhu', 'kualitas'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.sort_values('timestamp')
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching IoT data: {e}")
        return pd.DataFrame()

def get_ml_predictions(limit=100):
    """Get latest ML predictions"""
    try:
        connection = get_database_connection()
        if connection:
            cursor = connection.cursor()
            # Check if predictions table exists
            cursor.execute("SHOW TABLES LIKE 'predictions'")
            if cursor.fetchone():
                query = """
                SELECT timestamp, model_type, prediction_data 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
                data = cursor.fetchall()
                cursor.close()
                
                if data:
                    df = pd.DataFrame(data, columns=['timestamp', 'model_type', 'prediction_data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
            else:
                # Generate mock predictions for demo
                return generate_mock_predictions(limit)
    except Exception as e:
        st.warning(f"Using mock predictions: {e}")
        return generate_mock_predictions(limit)
    
    return pd.DataFrame()

def generate_mock_predictions(limit=100):
    """Generate mock ML predictions for demo"""
    timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='30S')
    models = ['cnn_prediction', 'lstm_prediction', 'hybrid_prediction']
    
    data = []
    for i, ts in enumerate(timestamps):
        for model in models:
            prediction = {
                'timestamp': ts,
                'model_type': model,
                'ph': 7.0 + np.random.normal(0, 0.3),
                'suhu': 25.0 + np.random.normal(0, 2.0),
                'kualitas': np.random.choice(['baik', 'buruk'], p=[0.7, 0.3]),
                'confidence': 0.8 + np.random.normal(0, 0.1)
            }
            data.append(prediction)
    
    return pd.DataFrame(data)

def create_ml_metrics_cards(predictions_df):
    """Create ML performance metrics cards"""
    if predictions_df.empty:
        return
    
    # Calculate metrics per model
    models = predictions_df['model_type'].unique()
    
    cols = st.columns(len(models))
    
    for i, model in enumerate(models):
        model_data = predictions_df[predictions_df['model_type'] == model]
        
        if not model_data.empty:
            avg_confidence = model_data['confidence'].mean() if 'confidence' in model_data.columns else 0.85
            prediction_count = len(model_data)
            last_prediction = model_data['timestamp'].max()
            
            with cols[i]:
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>🤖 {model.replace('_', ' ').title()}</h4>
                    <p><strong>Confidence:</strong> {avg_confidence:.1%}</p>
                    <p><strong>Predictions:</strong> {prediction_count}</p>
                    <p class="model-status">Last: {last_prediction.strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)

def create_realtime_predictions_chart(predictions_df):
    """Create real-time predictions comparison chart"""
    if predictions_df.empty:
        st.warning("No prediction data available")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('pH Predictions', 'Temperature Predictions', 'Quality Distribution', 'Confidence Levels'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    colors = {'cnn_prediction': '#1f77b4', 'lstm_prediction': '#ff7f0e', 'hybrid_prediction': '#2ca02c'}
    
    # pH predictions
    for model in predictions_df['model_type'].unique():
        model_data = predictions_df[predictions_df['model_type'] == model]
        if not model_data.empty and 'ph' in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['ph'],
                    name=f'{model} pH',
                    line=dict(color=colors.get(model, '#636EFA')),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
    
    # Temperature predictions
    for model in predictions_df['model_type'].unique():
        model_data = predictions_df[predictions_df['model_type'] == model]
        if not model_data.empty and 'suhu' in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['suhu'],
                    name=f'{model} Temp',
                    line=dict(color=colors.get(model, '#636EFA')),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
    
    # Quality distribution
    if 'kualitas' in predictions_df.columns:
        quality_counts = predictions_df['kualitas'].value_counts()
        fig.add_trace(
            go.Bar(
                x=quality_counts.index,
                y=quality_counts.values,
                name='Quality Distribution',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
    
    # Confidence levels
    if 'confidence' in predictions_df.columns:
        for model in predictions_df['model_type'].unique():
            model_data = predictions_df[predictions_df['model_type'] == model]
            if not model_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=model_data['timestamp'],
                        y=model_data['confidence'],
                        name=f'{model} Confidence',
                        line=dict(color=colors.get(model, '#636EFA')),
                        mode='lines+markers'
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        height=600,
        title_text="🧠 Real-time ML Model Predictions",
        showlegend=True
    )
    
    return fig

def create_iot_data_monitoring(iot_df):
    """Create IoT data monitoring chart"""
    if iot_df.empty:
        st.warning("No IoT data available")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('pH Levels', 'Temperature', 'Quality Status', 'Data Flow Rate'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # pH levels
    fig.add_trace(
        go.Scatter(
            x=iot_df['timestamp'],
            y=iot_df['ph'],
            name='IoT pH',
            line=dict(color='blue'),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(
            x=iot_df['timestamp'],
            y=iot_df['suhu'],
            name='IoT Temperature',
            line=dict(color='red'),
            mode='lines'
        ),
        row=1, col=2
    )
    
    # Quality distribution
    quality_counts = iot_df['kualitas'].value_counts()
    fig.add_trace(
        go.Bar(
            x=quality_counts.index,
            y=quality_counts.values,
            name='Quality Distribution',
            marker_color='green'
        ),
        row=2, col=1
    )
    
    # Data flow rate (records per minute)
    iot_df['minute'] = iot_df['timestamp'].dt.floor('min')
    flow_rate = iot_df.groupby('minute').size().reset_index(name='records_per_minute')
    
    fig.add_trace(
        go.Scatter(
            x=flow_rate['minute'],
            y=flow_rate['records_per_minute'],
            name='Data Flow Rate',
            line=dict(color='orange'),
            mode='lines+markers'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text="📡 IoT Data Monitoring",
        showlegend=True
    )
    
    return fig

# Main dashboard
def main():
    # Header
    st.title("🧠 Machine Learning Output Dashboard")
    st.markdown("**Real-time AI Model Predictions from IoT Sensor Data**")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (seconds)",
        [10, 30, 60, 120],
        index=1
    )
    
    data_limit = st.sidebar.slider("Data Points", 50, 1000, 200)
    prediction_limit = st.sidebar.slider("Prediction Points", 20, 200, 50)
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_resource.clear()
        st.rerun()
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    # Load data
    with st.spinner("Loading IoT data..."):
        iot_df = get_iot_data(data_limit)
    
    with st.spinner("Loading ML predictions..."):
        predictions_df = get_ml_predictions(prediction_limit)
    
    # Status indicators
    iot_status = "🟢 Connected" if not iot_df.empty else "🔴 No Data"
    ml_status = "🟢 Active" if not predictions_df.empty else "🔴 Inactive"
    
    st.sidebar.markdown(f"**IoT Data:** {iot_status}")
    st.sidebar.markdown(f"**ML Engine:** {ml_status}")
    st.sidebar.markdown(f"**Records:** {len(iot_df)}")
    st.sidebar.markdown(f"**Predictions:** {len(predictions_df)}")
    
    # Main content
    if not iot_df.empty or not predictions_df.empty:
        # ML Metrics Cards
        st.subheader("🤖 AI Model Performance")
        create_ml_metrics_cards(predictions_df)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧠 ML Predictions")
            if not predictions_df.empty:
                fig_predictions = create_realtime_predictions_chart(predictions_df)
                st.plotly_chart(fig_predictions, use_container_width=True)
            else:
                st.info("No ML predictions available yet")
        
        with col2:
            st.subheader("📡 IoT Data Feed")
            if not iot_df.empty:
                fig_iot = create_iot_data_monitoring(iot_df)
                st.plotly_chart(fig_iot, use_container_width=True)
            else:
                st.info("Waiting for IoT data...")
        
        # Data tables
        with st.expander("📊 Recent Data Tables"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Latest IoT Data")
                if not iot_df.empty:
                    st.dataframe(iot_df.tail(10), use_container_width=True)
                else:
                    st.info("No IoT data")
            
            with col2:
                st.subheader("Latest ML Predictions")
                if not predictions_df.empty:
                    display_df = predictions_df.tail(10).copy()
                    if 'prediction_data' in display_df.columns:
                        display_df = display_df.drop('prediction_data', axis=1)
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info("No predictions")
    
    else:
        st.warning("""
        🚨 **No Data Available**
        
        **Possible Solutions:**
        1. **Check IoT Connection:** Make sure IoT devices are sending data to MySQL
        2. **Run Data Setup:** `cd data_insert && python setup_initial_data.py`
        3. **Start ML Engine:** `python ml_engine.py`
        4. **Check Database:** Verify MySQL service is running in Laragon
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🕐 Current Time", datetime.now().strftime("%H:%M:%S"))
    
    with col2:
        st.metric("📊 Data Points", len(iot_df))
    
    with col3:
        st.metric("🧠 ML Models", len(predictions_df['model_type'].unique()) if not predictions_df.empty else 0)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
