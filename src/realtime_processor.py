#!/usr/bin/env python3
"""
Real-time Processing Engine untuk Sensor Data
Integrasi Database, Preprocessing, dan AI Models
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

from database_manager import DatabaseManager
from realtime_preprocessor import SensorDataPreprocessor
from ai_models import SensorAIModels

class RealTimeProcessor:
    """Engine untuk processing data sensor secara real-time"""
    
    def __init__(self, retrain_interval_minutes=60, prediction_interval_seconds=5):
        # Components
        self.db_manager = DatabaseManager()
        self.preprocessor = SensorDataPreprocessor()
        self.ai_models = None
        
        # Configuration
        self.retrain_interval = retrain_interval_minutes * 60  # Convert to seconds
        self.prediction_interval = prediction_interval_seconds
        self.min_training_samples = 200
        
        # State management
        self.last_retrain_time = 0
        self.last_prediction_time = 0
        self.is_running = False
        self.models_trained = False
        
        # Data buffers
        self.prediction_buffer = deque(maxlen=1000)
        self.performance_buffer = deque(maxlen=100)
        
        # Setup logging
        self.setup_logging()
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'training_count': 0,
            'average_prediction_time': 0,
            'last_training_time': None
        }
    
    def setup_logging(self):
        """Setup logging untuk monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/realtime_processor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self):
        """Inisialisasi sistem"""
        self.logger.info("🚀 Menginisialisasi Real-time Processing Engine...")
        
        # Connect to database
        if not self.db_manager.connect():
            raise Exception("Gagal koneksi ke database!")
        
        # Create tables
        self.db_manager.create_tables()
        
        # Check data availability
        data_count = self.db_manager.get_data_count()
        self.logger.info(f"📊 Total data dalam database: {data_count}")
        
        if data_count >= self.min_training_samples:
            # Initialize preprocessor
            await self.initialize_preprocessor()
            
            # Initialize AI models
            await self.initialize_ai_models()
            
            # Initial training
            await self.train_models()
        else:
            self.logger.warning(f"⚠️ Data tidak cukup untuk training ({data_count}/{self.min_training_samples})")
            self.logger.info("🔄 Menunggu data lebih banyak...")
        
        self.logger.info("✅ Inisialisasi selesai!")
    
    async def initialize_preprocessor(self):
        """Inisialisasi preprocessor dengan data yang ada"""
        self.logger.info("🔧 Menginisialisasi preprocessor...")
        
        # Ambil data untuk fitting scalers
        training_data = self.db_manager.get_data_for_training(hours=24)  # Data 24 jam terakhir
        
        if not training_data.empty:
            # Create features
            featured_data = self.preprocessor.create_features(training_data)
            
            # Fit scalers
            self.preprocessor.fit_scalers(featured_data)
            
            self.logger.info(f"✅ Preprocessor berhasil di-fit dengan {len(featured_data)} samples")
        else:
            self.logger.error("❌ Tidak ada data untuk fitting preprocessor!")
            raise Exception("Tidak cukup data untuk fitting preprocessor")
    
    async def initialize_ai_models(self):
        """Inisialisasi AI models"""
        self.logger.info("🤖 Menginisialisasi AI models...")
        
        # Ambil sample data untuk menentukan input shape
        sample_data = self.db_manager.get_latest_data(100)
        
        if not sample_data.empty:
            featured_data = self.preprocessor.create_features(sample_data)
            feature_names = self.preprocessor.get_feature_names(sample_data)
            
            # Determine input shape
            input_shape = (self.preprocessor.window_size, len(feature_names))
            
            # Initialize AI models
            self.ai_models = SensorAIModels(input_shape=input_shape, output_dim=3)
            
            # Build all models
            self.ai_models.build_cnn_model()
            self.ai_models.build_lstm_model()
            self.ai_models.build_hybrid_cnn_lstm_model()
            
            self.logger.info(f"✅ AI models berhasil dibuat dengan input shape: {input_shape}")
        else:
            self.logger.error("❌ Tidak ada data untuk inisialisasi AI models!")
            raise Exception("Tidak cukup data untuk inisialisasi AI models")
    
    async def train_models(self):
        """Training semua AI models"""
        self.logger.info("🎓 Memulai training AI models...")
        
        start_time = time.time()
        
        try:
            # Ambil data training
            training_data = self.db_manager.get_data_for_training(hours=48)  # 48 jam data
            
            if len(training_data) < self.min_training_samples:
                self.logger.warning(f"⚠️ Data training tidak cukup: {len(training_data)}/{self.min_training_samples}")
                return False
            
            # Preprocessing
            featured_data = self.preprocessor.create_features(training_data)
            transformed_data = self.preprocessor.transform_features(featured_data)
            
            # Prepare sequence data
            X, y = self.preprocessor.prepare_sequence_data(transformed_data)
            
            if X is None or len(X) < 50:
                self.logger.warning("⚠️ Tidak cukup sequence data untuk training")
                return False
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y[:, 2].astype(int)
            )
            
            self.logger.info(f"📊 Data training: {X_train.shape}, validation: {X_val.shape}")
            
            # Train semua model
            model_results = {}
            
            for model_type in ['CNN', 'LSTM', 'CNN_LSTM']:
                self.logger.info(f"🎯 Training {model_type} model...")
                
                try:
                    history = self.ai_models.train_model(
                        model_type=model_type,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        epochs=50,  # Kurangi epochs untuk training yang lebih cepat
                        batch_size=32,
                        verbose=0
                    )
                    
                    # Evaluate model
                    metrics = self.ai_models.evaluate_model(model_type, X_val, y_val)
                    model_results[model_type] = metrics
                    
                    # Save performance to database
                    self.db_manager.save_model_performance(model_type, metrics)
                    
                    self.logger.info(f"✅ {model_type} training selesai - Accuracy: {metrics['accuracy']:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error training {model_type}: {e}")
                    continue
            
            # Save models
            if model_results:
                self.ai_models.save_models()
                self.models_trained = True
                self.performance_stats['training_count'] += 1
                self.performance_stats['last_training_time'] = datetime.now()
                
                training_time = time.time() - start_time
                self.logger.info(f"✅ Training semua model selesai dalam {training_time:.2f} detik")
                self.logger.info(f"📊 Model results: {model_results}")
                
                return True
            else:
                self.logger.error("❌ Tidak ada model yang berhasil di-train!")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error selama training: {e}")
            return False
    
    async def make_prediction(self):
        """Buat prediksi berdasarkan data terbaru"""
        if not self.models_trained:
            self.logger.warning("⚠️ Model belum di-train, skip prediksi")
            return None
        
        start_time = time.time()
        
        try:
            # Ambil data terbaru
            latest_data = self.db_manager.get_latest_data(self.preprocessor.window_size + 10)
            
            if len(latest_data) < self.preprocessor.window_size:
                self.logger.warning(f"⚠️ Data tidak cukup untuk prediksi: {len(latest_data)}/{self.preprocessor.window_size}")
                return None
            
            # Prepare input
            X_input = self.preprocessor.prepare_realtime_input(latest_data)
            
            # Prediksi dengan semua model
            predictions = {}
            
            for model_type in ['CNN', 'LSTM', 'CNN_LSTM']:
                if self.ai_models.is_trained.get(model_type, False):
                    try:
                        reg_pred, clf_pred = self.ai_models.predict(model_type, X_input, return_proba=True)
                        
                        # Inverse transform untuk mendapatkan nilai asli
                        reg_inverse, clf_proba = self.preprocessor.inverse_transform_targets(
                            np.concatenate([reg_pred, clf_pred], axis=1)
                        )
                        
                        predictions[model_type] = {
                            'ph': reg_inverse[0, 0],
                            'suhu': reg_inverse[0, 1],
                            'kualitas': 'baik' if clf_proba[0, 0] > 0.5 else 'buruk',
                            'confidence': clf_proba[0, 0] if clf_proba[0, 0] > 0.5 else 1 - clf_proba[0, 0]
                        }
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error prediksi {model_type}: {e}")
                        continue
            
            # Ensemble prediction
            if predictions:
                try:
                    ensemble_reg, ensemble_clf = self.ai_models.create_ensemble_prediction(X_input)
                    
                    # Inverse transform
                    reg_inverse, clf_proba = self.preprocessor.inverse_transform_targets(
                        np.concatenate([ensemble_reg, ensemble_clf], axis=1)
                    )
                    
                    predictions['ENSEMBLE'] = {
                        'ph': reg_inverse[0, 0],
                        'suhu': reg_inverse[0, 1],
                        'kualitas': 'baik' if clf_proba[0, 0] > 0.5 else 'buruk',
                        'confidence': clf_proba[0, 0] if clf_proba[0, 0] > 0.5 else 1 - clf_proba[0, 0]
                    }
                    
                except Exception as e:
                    self.logger.error(f"❌ Error ensemble prediction: {e}")
            
            # Save predictions to database
            latest_sensor_id = latest_data['no'].iloc[-1] if 'no' in latest_data.columns else None
            
            for model_type, pred in predictions.items():
                self.db_manager.save_prediction(latest_sensor_id, model_type, pred)
            
            # Update performance stats
            prediction_time = time.time() - start_time
            self.performance_stats['total_predictions'] += 1
            self.performance_stats['successful_predictions'] += 1
            self.performance_stats['average_prediction_time'] = (
                (self.performance_stats['average_prediction_time'] * (self.performance_stats['total_predictions'] - 1) + prediction_time) /
                self.performance_stats['total_predictions']
            )
            
            # Add to buffer
            prediction_result = {
                'timestamp': datetime.now(),
                'predictions': predictions,
                'processing_time': prediction_time,
                'sensor_id': latest_sensor_id
            }
            
            self.prediction_buffer.append(prediction_result)
            
            self.logger.info(f"✅ Prediksi selesai dalam {prediction_time:.3f}s - Models: {list(predictions.keys())}")
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"❌ Error selama prediksi: {e}")
            self.performance_stats['total_predictions'] += 1
            return None
    
    async def run_realtime_loop(self):
        """Loop utama untuk processing real-time"""
        self.logger.info("🔄 Memulai real-time processing loop...")
        self.is_running = True
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check apakah perlu retrain
                if (current_time - self.last_retrain_time) >= self.retrain_interval:
                    self.logger.info("🔄 Waktu untuk retrain model...")
                    
                    data_count = self.db_manager.get_data_count()
                    if data_count >= self.min_training_samples:
                        await self.train_models()
                    
                    self.last_retrain_time = current_time
                
                # Check apakah perlu prediksi
                if (current_time - self.last_prediction_time) >= self.prediction_interval:
                    await self.make_prediction()
                    self.last_prediction_time = current_time
                
                # Monitor system health
                if self.performance_stats['total_predictions'] % 100 == 0 and self.performance_stats['total_predictions'] > 0:
                    await self.log_performance_stats()
                
                # Sleep untuk menghemat CPU
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Error dalam real-time loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def log_performance_stats(self):
        """Log statistik performa sistem"""
        stats = self.performance_stats.copy()
        
        success_rate = (stats['successful_predictions'] / stats['total_predictions'] * 100) if stats['total_predictions'] > 0 else 0
        
        # Get database stats
        db_stats = self.db_manager.get_realtime_stats()
        
        self.logger.info("📊 Performance Statistics:")
        self.logger.info(f"   Total Predictions: {stats['total_predictions']}")
        self.logger.info(f"   Success Rate: {success_rate:.2f}%")
        self.logger.info(f"   Avg Prediction Time: {stats['average_prediction_time']:.3f}s")
        self.logger.info(f"   Training Count: {stats['training_count']}")
        self.logger.info(f"   Last Training: {stats['last_training_time']}")
        
        if db_stats:
            self.logger.info(f"   DB Total Readings (1h): {db_stats.get('total_readings', 0)}")
            self.logger.info(f"   DB Avg pH (1h): {db_stats.get('avg_ph', 0):.2f}")
            self.logger.info(f"   DB Avg Suhu (1h): {db_stats.get('avg_suhu', 0):.2f}")
    
    def stop(self):
        """Stop real-time processing"""
        self.logger.info("🛑 Menghentikan real-time processing...")
        self.is_running = False
        self.db_manager.disconnect()
    
    def get_recent_predictions(self, limit=10):
        """Ambil prediksi terbaru"""
        return list(self.prediction_buffer)[-limit:] if self.prediction_buffer else []
    
    def get_performance_summary(self):
        """Ambil ringkasan performa sistem"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'models_trained': self.models_trained,
            'is_running': self.is_running,
            'buffer_size': len(self.prediction_buffer),
            'database_connected': self.db_manager.connection is not None
        }

# Testing dan demo
async def main():
    """Main function untuk testing"""
    processor = RealTimeProcessor(
        retrain_interval_minutes=30,  # Retrain setiap 30 menit
        prediction_interval_seconds=3  # Prediksi setiap 3 detik
    )
    
    try:
        # Initialize
        await processor.initialize()
        
        # Run for demo (30 seconds)
        print("🚀 Memulai real-time processing demo...")
        print("⏰ Demo akan berjalan selama 30 detik...")
        
        # Start processing in background
        processing_task = asyncio.create_task(processor.run_realtime_loop())
        
        # Monitor untuk 30 detik
        for i in range(30):
            await asyncio.sleep(1)
            
            if i % 10 == 0:  # Log setiap 10 detik
                recent_predictions = processor.get_recent_predictions(3)
                if recent_predictions:
                    print(f"📊 Recent predictions: {len(recent_predictions)} available")
                    latest = recent_predictions[-1]
                    print(f"   Latest prediction: {latest['predictions'].get('ENSEMBLE', 'N/A')}")
        
        print("⏹️ Demo selesai, menghentikan processor...")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo dihentikan oleh user")
    finally:
        processor.stop()
        print("✅ Processor berhasil dihentikan")

if __name__ == "__main__":
    asyncio.run(main())
