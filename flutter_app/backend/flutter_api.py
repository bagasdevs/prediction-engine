#!/usr/bin/env python3
"""
Flutter API Wrapper untuk Prediction Engine
Menghubungkan Flutter app dengan ML Engine yang sudah ada
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import time
import json
from datetime import datetime
import mysql.connector
import random
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from flutter_app.backend.ml_engine import MLEngine
    ML_ENGINE_AVAILABLE = True
    print("✅ ML Engine loaded successfully")
except ImportError as e:
    print(f"⚠️ ML Engine not available: {e}")
    ML_ENGINE_AVAILABLE = False
    
    # Fallback dummy ML engine
    class DummyMLEngine:
        def predict(self, ph, suhu):
            # Simple rule-based prediction
            if ph < 6.5 or ph > 8.5:
                return "Buruk", 0.85
            elif suhu < 20 or suhu > 30:
                return "Sedang", 0.70
            else:
                return "Baik", 0.90
    
    MLEngine = DummyMLEngine

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from flutter_app.backend.ml_engine import MLEngine
    from simulasi_data.simulasi import SensorDataSimulator
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    # Try local simulasi_data
    try:
        from simulasi_data.simulasi1 import SensorDataSimulator
    except ImportError:
        print("⚠️ Local simulator also not found")

app = Flask(__name__)
CORS(app)  # Enable CORS untuk Flutter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
ml_engine = None
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'sensor_data'
}

def get_db_connection():
    """Get database connection"""
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return None

def generate_prediction_data():
    """Generate sample prediction data"""
    # Generate random sensor data
    ph = round(random.uniform(4.0, 10.0), 2)
    suhu = round(random.uniform(15.0, 40.0), 2)
    
    # Simple prediction logic
    if (6.5 <= ph <= 7.5) and (20 <= suhu <= 30):
        kualitas = 'baik'
        confidence = round(random.uniform(85, 95), 1)
    elif (6.0 <= ph <= 8.0) and (18 <= suhu <= 35):
        kualitas = 'sedang'
        confidence = round(random.uniform(70, 85), 1)
    else:
        kualitas = 'buruk'
        confidence = round(random.uniform(60, 80), 1)
    
    return {
        'ph': ph,
        'suhu': suhu,
        'kualitas': kualitas,
        'confidence': confidence,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Prediction Engine API'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        if 'type' in data and data['type'] == 'sensor':
            # Handle sensor data prediction
            ph = float(data.get('ph', 7.0))
            suhu = float(data.get('suhu', 25.0))
            
            # Simple prediction logic based on ranges
            if (6.5 <= ph <= 7.5) and (20 <= suhu <= 30):
                prediction = 'baik'
                confidence = 90.0
                recommendation = 'Kondisi optimal, tidak perlu tindakan khusus'
            elif (6.0 <= ph <= 8.0) and (18 <= suhu <= 35):
                prediction = 'sedang'
                confidence = 75.0
                recommendation = 'Perlu monitoring lebih ketat'
            else:
                prediction = 'buruk'
                confidence = 85.0
                recommendation = 'Perlu tindakan segera untuk perbaikan'
            
            # Save prediction to database if possible
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO sensor_readings (ph, suhu, kualitas, timestamp) 
                        VALUES (%s, %s, %s, %s)
                    """, (ph, suhu, prediction, datetime.now()))
                    conn.commit()
                    cursor.close()
                    conn.close()
            except Exception as db_err:
                logger.warning(f"Database save failed: {db_err}")
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': confidence,
                'recommendation': recommendation,
                'input_data': {
                    'ph': ph,
                    'suhu': suhu
                },
                'timestamp': datetime.now().isoformat()
            })
        
        else:
            # Generate random prediction for testing
            pred_data = generate_prediction_data()
            return jsonify({
                'success': True,
                'prediction': pred_data['kualitas'],
                'confidence': pred_data['confidence'],
                'data': pred_data,
                'message': 'Sample prediction generated'
            })
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }), 400

@app.route('/simulate', methods=['POST'])
def simulate_data():
    """Generate and insert simulation data"""
    try:
        data = request.get_json()
        count = int(data.get('count', 10))
        
        generated_data = []
        conn = get_db_connection()
        
        if conn:
            cursor = conn.cursor()
            
            for i in range(count):
                pred_data = generate_prediction_data()
                
                # Insert to database
                cursor.execute("""
                    INSERT INTO sensor_readings (ph, suhu, kualitas, timestamp) 
                    VALUES (%s, %s, %s, %s)
                """, (pred_data['ph'], pred_data['suhu'], 
                     pred_data['kualitas'], datetime.now()))
                
                generated_data.append(pred_data)
            
            conn.commit()
            cursor.close()
            conn.close()
        
        return jsonify({
            'success': True,
            'message': f'{count} data points generated',
            'sample_data': generated_data[:5]  # Show first 5 samples
        })
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/recent-data', methods=['GET'])
def get_recent_data():
    """Get recent sensor data"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        data_list = []
        for row in results:
            data_list.append({
                'ph': float(row[0]),
                'suhu': float(row[1]),
                'kualitas': row[2],
                'timestamp': row[3].isoformat() if row[3] else None
            })
        
        return jsonify({
            'success': True,
            'data': data_list,
            'count': len(data_list)
        })
        
    except Exception as e:
        logger.error(f"Data retrieval error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/simulation/start', methods=['POST'])
def start_simulation():
    """Start automatic data simulation"""
    try:
        data = request.get_json() or {}
        interval = data.get('interval', 5)  # default 5 seconds
        
        # Start background simulation
        # For simplicity, we'll use threading
        import threading
        from simulasi_data.simulasi1 import SensorDataSimulator
        
        def run_background_simulation():
            simulator = SensorDataSimulator()
            simulator.run_simulation(interval=interval, verbose=False)
        
        # Start in background thread
        simulation_thread = threading.Thread(target=run_background_simulation, daemon=True)
        simulation_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Simulation started with {interval}s interval',
            'interval': interval
        })
        
    except Exception as e:
        logger.error(f"Simulation start error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/simulation/status', methods=['GET'])
def simulation_status():
    """Get simulation status"""
    try:
        # Check if there's recent data (last 30 seconds)
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM sensor_readings 
                WHERE timestamp > DATE_SUB(NOW(), INTERVAL 30 SECOND)
            """)
            recent_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            is_active = recent_count > 0
            
            return jsonify({
                'success': True,
                'simulation_active': is_active,
                'recent_records': recent_count,
                'message': 'Active' if is_active else 'Inactive'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Simulation status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/live-data', methods=['GET'])
def get_live_data():
    """Get live sensor data dengan timestamp info"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'error': 'Database connection failed'
            }), 500
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ph, suhu, kualitas, timestamp,
                   TIMESTAMPDIFF(SECOND, timestamp, NOW()) as seconds_ago
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        data_list = []
        for row in results:
            data_list.append({
                'ph': float(row[0]),
                'suhu': float(row[1]),
                'kualitas': row[2],
                'timestamp': row[3].isoformat() if row[3] else None,
                'seconds_ago': row[4] if row[4] is not None else 0,
                'is_recent': row[4] <= 10 if row[4] is not None else False  # Recent if < 10 seconds
            })
        
        # Get latest record for quick stats
        latest = data_list[0] if data_list else None
        
        return jsonify({
            'success': True,
            'data': data_list,
            'count': len(data_list),
            'latest': latest,
            'server_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Live data retrieval error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    print("🚀 Starting Flutter API Server...")
    print("📱 Flutter app can connect to: http://localhost:5000")
    print("🔗 Endpoints:")
    print("   - POST /predict (main prediction)")
    print("   - POST /simulate (generate data)")
    print("   - GET /recent-data (get recent data)")
    print("   - GET /health (health check)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
