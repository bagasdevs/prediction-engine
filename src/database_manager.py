#!/usr/bin/env python3
"""
Real-time Sensor Data Processing System
Database Configuration & Connection Manager
"""

import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

class DatabaseManager:
    """Mengelola koneksi dan operasi database"""
    
    def __init__(self, config=None):
        self.config = config or {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'sensor_data',
            'port': 3306
        }
        self.connection = None
        self.cursor = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging untuk monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Koneksi ke database"""
        try:
            print(f"🔍 Attempting connection with config: {self.config}")
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            print("✅ Database connection successful")
            self.logger.info("✅ Berhasil terhubung ke database")
            return True
        except Error as e:
            print(f"❌ MySQL Error: {e}")
            self.logger.error(f"❌ Error koneksi database: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            self.logger.error(f"❌ Unexpected error: {e}")
            return False
    
    def disconnect(self):
        """Tutup koneksi database"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        self.logger.info("✅ Koneksi database ditutup")
    
    def create_tables(self):
        """Buat tabel jika belum ada"""
        create_sensor_table = """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            no INT AUTO_INCREMENT PRIMARY KEY,
            ph DECIMAL(5,2) NOT NULL,
            suhu DECIMAL(5,2) NOT NULL,
            kualitas ENUM('baik', 'buruk') NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp),
            INDEX idx_no (no)
        )
        """
        
        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            sensor_id INT,
            model_type ENUM('CNN', 'LSTM', 'CNN_LSTM', 'ENSEMBLE') NOT NULL,
            predicted_ph DECIMAL(5,2),
            predicted_suhu DECIMAL(5,2),
            predicted_kualitas ENUM('baik', 'sedang', 'buruk'),
            confidence_score DECIMAL(5,4),
            actual_ph DECIMAL(5,2) NULL,
            actual_suhu DECIMAL(5,2) NULL,
            actual_kualitas ENUM('baik', 'sedang', 'buruk') NULL,
            accuracy DECIMAL(5,4) NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_sensor_id (sensor_id),
            INDEX idx_timestamp (timestamp)
        )
        """
        
        create_model_performance = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_type ENUM('CNN', 'LSTM', 'CNN_LSTM', 'ENSEMBLE') NOT NULL,
            accuracy DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            rmse_ph DECIMAL(6,4),
            rmse_suhu DECIMAL(6,4),
            training_samples INT,
            evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        try:
            self.cursor.execute(create_sensor_table)
            self.cursor.execute(create_predictions_table)
            self.cursor.execute(create_model_performance)
            self.connection.commit()
            self.logger.info("✅ Tabel database berhasil dibuat/diverifikasi")
            return True
        except Error as e:
            self.logger.error(f"❌ Error membuat tabel: {e}")
            return False
    
    def get_latest_data(self, limit=100):
        """Ambil data sensor terbaru"""
        try:
            query = """
            SELECT no, ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            self.cursor.execute(query, (limit,))
            data = self.cursor.fetchall()
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Error as e:
            self.logger.error(f"❌ Error mengambil data: {e}")
            return pd.DataFrame()
    
    def get_data_for_training(self, hours=24):
        """Ambil data untuk training model"""
        try:
            query = """
            SELECT no, ph, suhu, kualitas, timestamp 
            FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL %s HOUR
            ORDER BY timestamp ASC
            """
            self.cursor.execute(query, (hours,))
            data = self.cursor.fetchall()
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Error as e:
            self.logger.error(f"❌ Error mengambil data training: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, sensor_id, model_type, prediction_data):
        """Simpan hasil prediksi"""
        try:
            query = """
            INSERT INTO ml_predictions 
            (sensor_id, model_type, predicted_ph, predicted_suhu, 
             predicted_kualitas, confidence_score) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (
                sensor_id,
                model_type,
                prediction_data['ph'],
                prediction_data['suhu'], 
                prediction_data['kualitas'],
                prediction_data['confidence']
            ))
            self.connection.commit()
            return True
        except Error as e:
            self.logger.error(f"❌ Error menyimpan prediksi: {e}")
            return False
    
    def save_model_performance(self, model_type, metrics):
        """Simpan performa model"""
        try:
            query = """
            INSERT INTO model_performance 
            (model_type, accuracy, precision_score, recall_score, 
             f1_score, rmse_ph, rmse_suhu, training_samples) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.execute(query, (
                model_type,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['rmse_ph'],
                metrics['rmse_suhu'],
                metrics['training_samples']
            ))
            self.connection.commit()
            return True
        except Error as e:
            self.logger.error(f"❌ Error menyimpan performa model: {e}")
            return False
    
    def get_data_count(self):
        """Hitung total data dalam database"""
        try:
            query = "SELECT COUNT(*) as total FROM sensor_readings"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result['total'] if result else 0
        except Error as e:
            self.logger.error(f"❌ Error menghitung data: {e}")
            return 0
    
    def get_realtime_stats(self):
        """Statistik real-time dari database"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_readings,
                AVG(ph) as avg_ph,
                AVG(suhu) as avg_suhu,
                SUM(CASE WHEN kualitas = 'baik' THEN 1 ELSE 0 END) as good_quality,
                SUM(CASE WHEN kualitas = 'buruk' THEN 1 ELSE 0 END) as bad_quality,
                MAX(timestamp) as latest_reading
            FROM sensor_readings 
            WHERE timestamp >= NOW() - INTERVAL 1 HOUR
            """
            self.cursor.execute(query)
            return self.cursor.fetchone()
        except Error as e:
            self.logger.error(f"❌ Error mengambil statistik: {e}")
            return {}

    def fetch_data(self, query, params=None):
        """Execute query dan return DataFrame - untuk kompatibilitas dengan AI models"""
        try:
            self.cursor.execute(query, params or ())
            data = self.cursor.fetchall()
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Error as e:
            self.logger.error(f"❌ Error executing query: {e}")
            return pd.DataFrame()
    
    def insert_sensor_data(self, ph, suhu, kualitas):
        """Insert data sensor baru"""
        try:
            query = """
            INSERT INTO sensor_readings (ph, suhu, kualitas) 
            VALUES (%s, %s, %s)
            """
            self.cursor.execute(query, (ph, suhu, kualitas))
            self.connection.commit()
            return self.cursor.lastrowid
        except Error as e:
            self.logger.error(f"❌ Error inserting sensor data: {e}")
            return None

if __name__ == "__main__":
    # Test database connection
    db = DatabaseManager()
    if db.connect():
        db.create_tables()
        print("✅ Database setup berhasil!")
        print(f"📊 Total data: {db.get_data_count()}")
        
        # Test ambil data
        latest_data = db.get_latest_data(10)
        if not latest_data.empty:
            print(f"📈 Data terbaru:")
            print(latest_data.head())
        
        db.disconnect()
    else:
        print("❌ Gagal koneksi ke database!")
