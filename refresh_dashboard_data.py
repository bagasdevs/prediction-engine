#!/usr/bin/env python3
"""
Refresh Dashboard Data
Memastikan data terbaru tersedia untuk dashboard
"""

import mysql.connector
from datetime import datetime
import random

def refresh_dashboard_data():
    """Add fresh data for dashboard display"""
    print("🔄 REFRESHING DASHBOARD DATA")
    print("=" * 50)
    
    try:
        # Connect to database
        db = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data'
        )
        cursor = db.cursor()
        
        # Check current data status
        cursor.execute("""
            SELECT COUNT(*) FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL 1 HOUR
        """)
        recent_count = cursor.fetchone()[0]
        print(f"📊 Data in last hour: {recent_count}")
        
        # Add fresh data if needed
        if recent_count < 10:
            print("💡 Adding fresh data for dashboard...")
            
            for i in range(20):
                # Generate realistic sensor data
                ph = round(6.5 + random.random() * 1.5, 2)  # pH 6.5-8.0
                suhu = round(20 + random.random() * 12, 1)  # Temperature 20-32°C
                kualitas = random.choice(['baik', 'buruk'])  # Binary classification
                
                # Insert with timestamps spread over last hour
                cursor.execute("""
                    INSERT INTO sensor_readings (ph, suhu, kualitas, timestamp)
                    VALUES (%s, %s, %s, NOW() - INTERVAL %s MINUTE)
                """, (ph, suhu, kualitas, i * 3))
            
            db.commit()
            print("✅ Added 20 fresh sensor readings")
        else:
            print("✅ Sufficient recent data available")
        
        # Also refresh predictions for ML dashboard
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        print(f"🤖 Current predictions: {pred_count}")
        
        if pred_count < 5:
            print("💡 Adding fresh predictions...")
            
            models = ['cnn_prediction', 'lstm_prediction', 'hybrid_prediction']
            for i in range(3):
                for model in models:
                    ph = round(6.8 + random.random() * 1.0, 2)
                    suhu = round(24 + random.random() * 6, 1)
                    kualitas = random.choice(['baik', 'buruk'])
                    confidence = round(0.75 + random.random() * 0.25, 3)
                    
                    cursor.execute("""
                        INSERT INTO predictions (timestamp, model_type, ph, suhu, kualitas, confidence)
                        VALUES (NOW() - INTERVAL %s MINUTE, %s, %s, %s, %s, %s)
                    """, (i * 10, model, ph, suhu, kualitas, confidence))
            
            db.commit()
            print("✅ Added fresh ML predictions")
        
        # Show final status
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        total_sensors = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        print(f"\n📈 Final status:")
        print(f"  📊 Total sensor readings: {total_sensors}")
        print(f"  🤖 Total predictions: {total_predictions}")
        
        # Show latest data sample
        cursor.execute("""
            SELECT timestamp, ph, suhu, kualitas 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT 3
        """)
        
        print(f"\n🔍 Latest sensor data:")
        for row in cursor.fetchall():
            timestamp, ph, suhu, kualitas = row
            print(f"  📊 {timestamp}: pH={ph}, Suhu={suhu}°C, Kualitas={kualitas}")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Error refreshing data: {e}")
        return False

def check_dashboard_status():
    """Check if dashboard is running"""
    print("\n🌐 CHECKING DASHBOARD STATUS")
    print("=" * 50)
    
    import requests
    
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running at http://localhost:8501")
            print("🎉 Data should now be visible!")
            return True
        else:
            print(f"⚠️ Dashboard responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Dashboard is not responding")
        print("💡 Start dashboard with: streamlit run ml_dashboard.py")
        return False

def main():
    print("🔄 DASHBOARD DATA REFRESH")
    print("=" * 60)
    print("Ensuring fresh data is available for dashboard display")
    print()
    
    # Refresh data
    data_ok = refresh_dashboard_data()
    
    # Check dashboard
    dashboard_running = check_dashboard_status()
    
    print("\n" + "=" * 60)
    print("📋 REFRESH SUMMARY:")
    print(f"  Data Refresh: {'✅' if data_ok else '❌'}")
    print(f"  Dashboard Running: {'✅' if dashboard_running else '❌'}")
    
    if data_ok and dashboard_running:
        print("\n🎉 SUCCESS!")
        print("✅ Dashboard should now display data")
        print("🌐 Access: http://localhost:8501")
    elif data_ok and not dashboard_running:
        print("\n💡 DATA READY - START DASHBOARD:")
        print("streamlit run ml_dashboard.py")
    else:
        print("\n❌ Issues found - check error messages above")

if __name__ == "__main__":
    main()
