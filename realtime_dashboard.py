#!/usr/bin/env python3
"""
Real-time Dashboard untuk Sensor Data AI System
Monitoring real-time dengan CNN, LSTM, dan Hybrid models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database_manager import DatabaseManager
    from src.realtime_processor import RealTimeProcessor
    from ai_system_manager import AISystemManager
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("💡 Make sure all required files are in the src/ directory")
    st.stop()

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🤖 Real-time Sensor AI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'ai_manager' not in st.session_state:
    st.session_state.ai_manager = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = False
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []

# Helper functions
@st.cache_data(ttl=30)
def get_database_stats():
    """Get database statistics with caching"""
    try:
        db = DatabaseManager()
        if db.connect():
            stats = db.get_realtime_stats()
            db.disconnect()
            st.session_state.connection_status = True
            return stats
        else:
            st.session_state.connection_status = False
            return {}
    except Exception as e:
        st.session_state.connection_status = False
        error_msg = f"Database stats error: {str(e)}"
        if error_msg not in st.session_state.error_messages:
            st.session_state.error_messages.append(error_msg)
        return {}

@st.cache_data(ttl=20)
def get_recent_data(hours=1):
    """Get recent sensor data with caching"""
    try:
        db = DatabaseManager()
        if db.connect():
            # Modified to handle column names correctly
            query = f"""
            SELECT no, ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL {hours} HOUR
            ORDER BY timestamp DESC
            """
            result = db.fetch_data(query)
            db.disconnect()
            
            if not result.empty and len(result.columns) >= 5:
                # Ensure column names are correct
                result.columns = ['no', 'ph', 'suhu', 'kualitas', 'timestamp']
                return result
            else:
                return pd.DataFrame()
    except Exception as e:
        error_msg = f"Recent data error: {str(e)}"
        if error_msg not in st.session_state.error_messages:
            st.session_state.error_messages.append(error_msg)
        return pd.DataFrame()

def test_database_connection():
    """Test database connection"""
    try:
        db = DatabaseManager()
        if db.connect():
            # Test query
            result = db.fetch_data("SELECT COUNT(*) as count FROM sensor_readings")
            db.disconnect()
            return True, result.iloc[0]['count'] if not result.empty else 0
        else:
            return False, 0
    except Exception as e:
        return False, str(e)

def create_realtime_chart(data, title, y_col, color='blue'):
    """Create real-time line chart"""
    if data.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data[y_col],
        mode='lines+markers',
        name=y_col.title(),
        line=dict(color=color, width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_col.title(),
        height=300,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_prediction_comparison_chart(recent_predictions):
    """Create prediction comparison chart"""
    if not recent_predictions:
        # Create empty chart with message
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(height=400)
        return fig
    
    # Extract data for comparison (simplified version)
    timestamps = []
    ph_values = []
    suhu_values = []
    confidence_values = []
    
    for pred in recent_predictions[-10:]:  # Last 10 predictions
        timestamps.append(pred.get('timestamp', datetime.now()))
        ph_values.append(pred.get('ph_predicted', 7.0))
        suhu_values.append(pred.get('suhu_predicted', 25.0))
        confidence_values.append(pred.get('confidence', 0.5))
    
    # Create simple line chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('pH Predictions', 'Temperature Predictions', 'Confidence Scores'),
        vertical_spacing=0.1
    )
    
    # pH predictions
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ph_values,
        mode='lines+markers',
        name='pH Prediction',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # Temperature predictions
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=suhu_values,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='red', width=2),
        showlegend=False
    ), row=2, col=1)
    
    # Confidence scores
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidence_values,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='green', width=2),
        showlegend=False
    ), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def create_model_performance_heatmap():
    """Create model performance heatmap"""
    try:
        db = DatabaseManager()
        if db.connect():
            # Check if performance table exists first
            check_table_query = """
            SELECT COUNT(*) as count FROM information_schema.tables 
            WHERE table_schema = 'sensor_data' AND table_name = 'model_performance'
            """
            result = db.fetch_data(check_table_query)
            
            if result.empty or result.iloc[0]['count'] == 0:
                db.disconnect()
                # Return empty figure with message
                fig = go.Figure()
                fig.add_annotation(
                    text="No model performance data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16)
                )
                fig.update_layout(height=400, title="Model Performance Heatmap")
                return fig
            
            query = """
            SELECT model_type, accuracy, precision_score, recall_score, f1_score, rmse_ph, rmse_suhu
            FROM model_performance 
            ORDER BY evaluation_timestamp DESC 
            LIMIT 20
            """
            result = db.fetch_data(query)
            db.disconnect()
            
            if not result.empty:
                # Ensure we have the right columns
                expected_cols = ['model_type', 'accuracy', 'precision_score', 'recall_score', 'f1_score', 'rmse_ph', 'rmse_suhu']
                if all(col in result.columns for col in expected_cols):
                    # Normalize metrics untuk heatmap
                    metrics = ['accuracy', 'precision_score', 'recall_score', 'f1_score']
                    df_metrics = result.groupby('model_type')[metrics].mean()
                    
                    if not df_metrics.empty:
                        fig = px.imshow(
                            df_metrics.T,
                            labels=dict(x="Model Type", y="Metrics", color="Score"),
                            x=df_metrics.index,
                            y=metrics,
                            color_continuous_scale="Viridis",
                            aspect="auto"
                        )
                        
                        fig.update_layout(title="Model Performance Heatmap", height=400)
                        return fig
    except Exception as e:
        error_msg = f"Performance heatmap error: {str(e)}"
        if error_msg not in st.session_state.error_messages:
            st.session_state.error_messages.append(error_msg)
    
    # Return empty figure as fallback
    fig = go.Figure()
    fig.add_annotation(
        text="Performance data not available",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(height=400, title="Model Performance Heatmap")
    return fig

# Main dashboard
def main():
    # Header
    st.title("🤖 Real-time Sensor AI Dashboard")
    st.markdown("**Advanced CNN, LSTM & Hybrid Models for Sensor Data Prediction**")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    # Refresh interval
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (seconds)",
        [3, 5, 10, 15, 30],
        index=1
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Manual Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    # System status
    st.sidebar.subheader("🔧 System Status")
    
    # Database status
    connection_ok, db_info = test_database_connection()
    if connection_ok:
        st.sidebar.success("✅ Database: Connected")
        if isinstance(db_info, int):
            st.sidebar.metric("Total Records", db_info)
    else:
        st.sidebar.error("❌ Database: Disconnected")
        if isinstance(db_info, str):
            st.sidebar.error(f"Error: {db_info}")
    
    # Error messages
    if st.session_state.error_messages:
        st.sidebar.subheader("⚠️ Recent Errors")
        for error in st.session_state.error_messages[-3:]:  # Show last 3 errors
            st.sidebar.error(error)
        
        if st.sidebar.button("🧹 Clear Errors"):
            st.session_state.error_messages = []
            st.rerun()
    
    # AI System status
    if st.session_state.ai_manager:
        st.sidebar.success("✅ AI System: Running")
        try:
            status = st.session_state.ai_manager.get_system_status()
            st.sidebar.metric("Predictions Made", status.get('prediction_count', 0))
        except:
            st.sidebar.warning("⚠️ AI System: Status Unknown")
    else:
        st.sidebar.warning("⚠️ AI System: Not Started")
        if st.sidebar.button("🚀 Start AI System"):
            with st.spinner("Starting AI System Manager..."):
                try:
                    st.session_state.ai_manager = AISystemManager()
                    st.sidebar.success("✅ AI System Started!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {e}")
                    st.session_state.error_messages.append(f"AI System start error: {str(e)}")
    
    # Main content
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Current statistics
    with col1:
        st.subheader("📊 Current Statistics")
        db_stats = get_database_stats()
        if db_stats and st.session_state.connection_status:
            st.metric("Average pH (1h)", f"{db_stats.get('avg_ph', 0):.2f}")
            st.metric("Average Temperature (1h)", f"{db_stats.get('avg_suhu', 0):.1f}°C")
            
            good_quality = db_stats.get('good_quality', 0)
            bad_quality = db_stats.get('bad_quality', 0)
            total_quality = good_quality + bad_quality
            
            if total_quality > 0:
                quality_pct = (good_quality / total_quality) * 100
                st.metric("Good Quality %", f"{quality_pct:.1f}%")
        else:
            st.warning("📊 No statistics available")
            st.info("Make sure simulator is running")
    
    with col2:
        st.subheader("🤖 AI Model Status")
        
        # Mock model performance (replace with actual data)
        models_status = {
            'CNN': {'accuracy': 0.85, 'status': '✅ Active'},
            'LSTM': {'accuracy': 0.82, 'status': '✅ Active'},
            'CNN-LSTM': {'accuracy': 0.88, 'status': '✅ Active'},
            'Ensemble': {'accuracy': 0.90, 'status': '✅ Active'}
        }
        
        for model, info in models_status.items():
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.write(f"**{model}**")
            with col_b:
                st.write(f"{info['accuracy']:.3f} {info['status']}")
    
    with col3:
        st.subheader("⚡ System Performance")
        
        if st.session_state.ai_manager:
            try:
                status = st.session_state.ai_manager.get_system_status()
                
                st.metric("System Running", "✅ Yes" if status.get('system_running', False) else "❌ No")
                st.metric("Predictions Made", status.get('prediction_count', 0))
                
                if status.get('ai_health'):
                    ai_health = status['ai_health']
                    st.metric("Models Loaded", f"{ai_health.get('models_loaded', 0)}/3")
                    
            except Exception as e:
                st.error(f"Error getting system status: {str(e)}")
        else:
            st.info("🚀 AI System not running")
            st.markdown("Click 'Start AI System' in sidebar")
    
    # Charts section
    st.markdown("---")
    st.subheader("📈 Real-time Data Visualization")
    
    # Get recent data
    recent_data = get_recent_data(hours=2)
    
    if not recent_data.empty:
        # Time series charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            ph_chart = create_realtime_chart(recent_data, "pH Levels (2 hours)", "ph", "blue")
            st.plotly_chart(ph_chart, use_container_width=True)
        
        with chart_col2:
            temp_chart = create_realtime_chart(recent_data, "Temperature (2 hours)", "suhu", "red")
            st.plotly_chart(temp_chart, use_container_width=True)
        
        # Quality distribution
        quality_col1, quality_col2 = st.columns(2)
        
        with quality_col1:
            if 'kualitas' in recent_data.columns:
                quality_counts = recent_data['kualitas'].value_counts()
                
                fig_pie = px.pie(
                    values=quality_counts.values,
                    names=quality_counts.index,
                    title="Quality Distribution (2 hours)",
                    color_discrete_map={'baik': 'green', 'buruk': 'red'}
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with quality_col2:
            # Recent readings table
            st.subheader("🔍 Recent Readings")
            recent_display = recent_data.tail(10)[['timestamp', 'ph', 'suhu', 'kualitas']].copy()
            recent_display['timestamp'] = recent_display['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(recent_display, use_container_width=True)
    
    else:
        st.warning("⚠️ No recent data available. Make sure the simulator is running.")
        st.info("💡 Start the simulator with: `python simulasi.py`")
    
    # AI Predictions section
    st.markdown("---")
    st.subheader("🔮 AI Model Predictions")
    
    if st.session_state.ai_manager:
        try:
            # Try to get recent predictions
            predictions = st.session_state.ai_manager.predict_realtime_batch(limit=10)
            
            if predictions:
                # Prediction comparison chart
                pred_chart = create_prediction_comparison_chart(predictions)
                st.plotly_chart(pred_chart, use_container_width=True)
                
                # Latest predictions table
                st.subheader("📋 Latest Predictions")
                
                # Create simple predictions table
                pred_df_data = []
                for i, pred in enumerate(predictions[-5:]):  # Last 5 predictions
                    pred_df_data.append({
                        '#': i+1,
                        'Timestamp': pred.get('timestamp_actual', 'N/A'),
                        'Predicted pH': f"{pred.get('ph_predicted', 0):.2f}",
                        'Actual pH': f"{pred.get('actual_ph', 0):.2f}",
                        'Predicted Temp': f"{pred.get('suhu_predicted', 0):.1f}°C",
                        'Actual Temp': f"{pred.get('actual_suhu', 0):.1f}°C",
                        'Confidence': f"{pred.get('confidence', 0):.3f}"
                    })
                
                if pred_df_data:
                    pred_df = pd.DataFrame(pred_df_data)
                    st.dataframe(pred_df, use_container_width=True)
            else:
                st.info("🔄 Waiting for predictions...")
                st.info("💡 Make sure there's sufficient data in the database")
                
        except Exception as e:
            st.error(f"❌ Error getting predictions: {str(e)}")
            st.session_state.error_messages.append(f"Prediction error: {str(e)}")
    else:
        st.info("🚀 Start the AI System to see predictions")
        st.info("💡 Use the sidebar to start the AI System Manager")
    
    # Model performance section
    st.markdown("---")
    st.subheader("📊 Model Performance Analysis")
    
    perf_heatmap = create_model_performance_heatmap()
    if perf_heatmap.data:
        st.plotly_chart(perf_heatmap, use_container_width=True)
    else:
        st.info("📈 No performance data available yet")
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**🤖 Real-time Sensor AI System**")
        st.markdown("Built with CNN, LSTM & Hybrid Models")
    
    with footer_col2:
        st.markdown(f"**📅 Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    with footer_col3:
        if st.button("📥 Download Data"):
            if not recent_data.empty:
                csv = recent_data.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Auto refresh logic
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.session_state.last_update = datetime.now()
        st.rerun()

if __name__ == "__main__":
    main()
