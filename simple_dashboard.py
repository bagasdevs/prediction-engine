#!/usr/bin/env python3
"""
Simple Stock Market Dashboard - Demo Version
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="Stock Market Prediction Engine",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("📈 Stock Market Prediction Engine")
st.markdown("**Demo Version - Simple Dashboard**")
st.markdown("---")

# Sidebar
st.sidebar.header("🎛️ Controls")
stock_symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    ["BIPA", "GOOGL", "MSFT", "TSLA", "AMZN", "META"]
)

prediction_days = st.sidebar.slider("Prediction Days", 1, 100, 7)

# Generate mock data
@st.cache_data
def generate_mock_data(symbol, days=100):
    """Generate mock stock data"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), 
        end=datetime.now(), 
        freq='D'
    )
    
    # Simulate stock price with random walk
    np.random.seed(hash(symbol) % 1000)
    base_price = np.random.uniform(100, 300)
    
    prices = [base_price]
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 2)
        new_price = max(prices[-1] + change, 10)  # Minimum $10
        prices.append(new_price)
    
    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

# Generate prediction data
@st.cache_data
def generate_predictions(current_price, days):
    """Generate mock predictions"""
    np.random.seed(42)
    predictions = []
    
    for i in range(days):
        # Simulate prediction with some randomness
        change = np.random.normal(0.5, 1.5)  # Slight upward bias
        predicted_price = current_price + change
        confidence = np.random.uniform(0.6, 0.9)
        
        predictions.append({
            'Day': i + 1,
            'Predicted_Price': predicted_price,
            'Confidence': confidence,
            'Signal': 'BUY' if change > 0 else 'SELL'
        })
        current_price = predicted_price
    
    return pd.DataFrame(predictions)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 {stock_symbol} Price History")
    
    # Get mock data
    historical_data = generate_mock_data(stock_symbol)
    current_price = historical_data['Price'].iloc[-1]
    
    # Create price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data['Date'],
        y=historical_data['Price'],
        mode='lines',
        name=f'{stock_symbol} Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{stock_symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Current Stats")
    
    # Display current stats
    st.metric("Current Price", f"${current_price:.2f}")
    st.metric("Today's Change", f"+${np.random.uniform(0.5, 3):.2f}", delta="2.1%")
    st.metric("Volume", f"{np.random.randint(1000000, 5000000):,}")
    
    # System status
    st.subheader("🔧 System Status")
    st.success("✅ Data Feed: Active")
    st.success("✅ ML Models: Loaded")
    st.success("✅ Predictions: Ready")

# Predictions section
st.subheader(f"🔮 Predictions for Next {prediction_days} Days")

predictions_df = generate_predictions(current_price, prediction_days)

col3, col4 = st.columns([2, 1])

with col3:
    # Prediction chart
    fig2 = go.Figure()
    
    # Historical (last 10 days)
    recent_data = historical_data.tail(10)
    fig2.add_trace(go.Scatter(
        x=recent_data['Date'],
        y=recent_data['Price'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Predictions
    future_dates = pd.date_range(
        start=historical_data['Date'].iloc[-1] + timedelta(days=1),
        periods=prediction_days,
        freq='D'
    )
    
    fig2.add_trace(go.Scatter(
        x=future_dates,
        y=predictions_df['Predicted_Price'],
        mode='lines+markers',
        name='Predictions',
        line=dict(color='red', dash='dash')
    ))
    
    fig2.update_layout(
        title="Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    st.subheader("📋 Prediction Summary")
    
    avg_confidence = predictions_df['Confidence'].mean()
    buy_signals = len(predictions_df[predictions_df['Signal'] == 'BUY'])
    
    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    st.metric("Buy Signals", f"{buy_signals}/{len(predictions_df)}")
    
    # Show predictions table
    st.dataframe(
        predictions_df[['Day', 'Predicted_Price', 'Confidence', 'Signal']], 
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("**Stock Market Prediction Engine Demo** | Built with Streamlit & Python")

# Auto-refresh every 30 seconds
time.sleep(0.1)
st.rerun()
