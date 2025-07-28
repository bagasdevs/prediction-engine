#!/usr/bin/env python3
"""
SQLite Database Manager untuk Real-time Sensor AI System
Alternative ke MySQL yang lebih mudah setup
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import threading
import json

class SQLiteDatabaseManager:
    """SQLite Database Manager untuk sensor data"""
    
    def __init__(self, db_path="sensor_data.db"):
        self.db_path = db_path
        self.connection = None
        self.health_stats = {
            'connection_status': 'disconnected',
            'last_query_time': None,
            'total_queries': 0,
            'avg_query_time': 0,
            'total_records': 0,
            'last_health_check': None
        }
        
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create table if not exists
            self.create_table()
            
            self.health_stats['connection_status'] = 'connected'
            self.health_stats['last_health_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"✅ Connected to SQLite database: {self.db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.health_stats['connection_status'] = 'failed'
            return False
    
    def create_table(self):
        """Create sensor_data table"""
        try:
            cursor = self.connection.cursor()
            
            # Create main sensor data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sensor_data (
                    no INTEGER PRIMARY KEY AUTOINCREMENT,
                    ph REAL NOT NULL,
                    suhu REAL NOT NULL,
                    kualitas TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    ph_pred REAL,
                    suhu_pred REAL,
                    kualitas_pred TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    training_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.connection.commit()
            print("✅ Database tables created/verified")
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise
    
    def insert_sensor_data(self, ph, suhu, kualitas):
        """Insert sensor data"""
        try:
            start_time = time.time()
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO sensor_data (ph, suhu, kualitas)
                VALUES (?, ?, ?)
            ''', (ph, suhu, kualitas))
            
            self.connection.commit()
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            return cursor.lastrowid
            
        except Exception as e:
            print(f"❌ Error inserting data: {e}")
            return None
    
    def get_data_for_training(self, hours=24):
        """Get data for training"""
        try:
            start_time = time.time()
            
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours)
            
            query = '''
                SELECT no, ph, suhu, kualitas, timestamp 
                FROM sensor_data 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            '''
            
            df = pd.read_sql_query(query, self.connection, params=(time_threshold,))
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            print(f"✅ Retrieved {len(df)} records for training")
            return df
            
        except Exception as e:
            print(f"❌ Error getting training data: {e}")
            return pd.DataFrame()
    
    def get_latest_data(self, limit=100):
        """Get latest sensor data"""
        try:
            start_time = time.time()
            
            query = '''
                SELECT no, ph, suhu, kualitas, timestamp 
                FROM sensor_data 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, self.connection, params=(limit,))
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting latest data: {e}")
            return pd.DataFrame()
    
    def store_prediction(self, model_type, ph_pred, suhu_pred, kualitas_pred, confidence=0.0):
        """Store prediction results"""
        try:
            start_time = time.time()
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (model_type, ph_pred, suhu_pred, kualitas_pred, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_type, ph_pred, suhu_pred, kualitas_pred, confidence))
            
            self.connection.commit()
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            return cursor.lastrowid
            
        except Exception as e:
            print(f"❌ Error storing prediction: {e}")
            return None
    
    def store_model_performance(self, model_type, accuracy, loss, training_time):
        """Store model performance metrics"""
        try:
            start_time = time.time()
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (model_type, accuracy, loss, training_time)
                VALUES (?, ?, ?, ?)
            ''', (model_type, accuracy, loss, training_time))
            
            self.connection.commit()
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            return cursor.lastrowid
            
        except Exception as e:
            print(f"❌ Error storing model performance: {e}")
            return None
    
    def get_predictions(self, model_type=None, limit=100):
        """Get prediction history"""
        try:
            start_time = time.time()
            
            if model_type:
                query = '''
                    SELECT * FROM predictions 
                    WHERE model_type = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, self.connection, params=(model_type, limit))
            else:
                query = '''
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                '''
                df = pd.read_sql_query(query, self.connection, params=(limit,))
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting predictions: {e}")
            return pd.DataFrame()
    
    def get_model_performance_history(self, model_type=None):
        """Get model performance history"""
        try:
            start_time = time.time()
            
            if model_type:
                query = '''
                    SELECT * FROM model_performance 
                    WHERE model_type = ?
                    ORDER BY timestamp DESC
                '''
                df = pd.read_sql_query(query, self.connection, params=(model_type,))
            else:
                query = '''
                    SELECT * FROM model_performance 
                    ORDER BY timestamp DESC
                '''
                df = pd.read_sql_query(query, self.connection)
            
            # Update stats
            query_time = time.time() - start_time
            self.update_query_stats(query_time)
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting performance history: {e}")
            return pd.DataFrame()
    
    def update_query_stats(self, query_time):
        """Update query statistics"""
        self.health_stats['total_queries'] += 1
        self.health_stats['last_query_time'] = query_time
        
        # Calculate rolling average
        if self.health_stats['avg_query_time'] == 0:
            self.health_stats['avg_query_time'] = query_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.health_stats['avg_query_time'] = (
                alpha * query_time + (1 - alpha) * self.health_stats['avg_query_time']
            )
    
    def get_health_stats(self):
        """Get database health statistics"""
        try:
            # Update total records count
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            self.health_stats['total_records'] = cursor.fetchone()[0]
            
            self.health_stats['last_health_check'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return self.health_stats.copy()
            
        except Exception as e:
            print(f"❌ Error getting health stats: {e}")
            return self.health_stats.copy()
    
    def disconnect(self):
        """Disconnect from database"""
        try:
            if self.connection:
                self.connection.close()
                self.health_stats['connection_status'] = 'disconnected'
                print("✅ Database disconnected")
        except Exception as e:
            print(f"❌ Error disconnecting: {e}")
    
    def __del__(self):
        """Destructor"""
        self.disconnect()

def test_database():
    """Test database functionality"""
    print("🧪 Testing SQLite Database Manager...")
    
    # Initialize database
    db = SQLiteDatabaseManager("test_sensor_data.db")
    
    # Connect
    if not db.connect():
        print("❌ Database test failed!")
        return False
    
    # Insert test data
    print("📝 Inserting test data...")
    for i in range(10):
        ph = np.random.normal(7, 0.5)
        suhu = np.random.normal(25, 3)
        kualitas = 'baik' if (6.5 <= ph <= 7.5) and (20 <= suhu <= 30) else 'buruk'
        
        db.insert_sensor_data(ph, suhu, kualitas)
    
    # Get data
    print("📊 Retrieving data...")
    data = db.get_latest_data(5)
    print(f"Retrieved {len(data)} records")
    print(data.head())
    
    # Store test prediction
    print("🔮 Storing test prediction...")
    db.store_prediction('CNN', 7.2, 25.5, 'baik', 0.95)
    
    # Get predictions
    predictions = db.get_predictions(limit=5)
    print(f"Predictions: {len(predictions)} records")
    
    # Health stats
    print("📈 Health Statistics:")
    stats = db.get_health_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    db.disconnect()
    os.remove("test_sensor_data.db")
    
    print("✅ Database test completed successfully!")
    return True

if __name__ == "__main__":
    test_database()
