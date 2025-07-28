#!/usr/bin/env python3
"""
AI System Manager - Koordinasi semua komponen sistem AI sensor real-time
"""

import time
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.ai_models import SensorAIModels
from src.realtime_preprocessor import SensorDataPreprocessor
from src.database_manager import DatabaseManager
import logging
import schedule

class AISystemManager:
    """Manager untuk koordinasi semua komponen AI system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.is_running = False
        self.auto_retrain_enabled = True
        
        # Initialize components
        self.db_manager = None
        self.preprocessor = None
        self.ai_models = None
        
        # System status
        self.last_training = None
        self.prediction_count = 0
        self.system_errors = []
        
        self._initialize_system()
    
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_system_manager.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_system(self):
        """Initialize semua komponen sistem"""
        try:
            self.logger.info("🚀 Initializing AI System Manager...")
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            self.logger.info("✅ Database manager initialized")
            
            # Initialize preprocessor
            self.preprocessor = SensorDataPreprocessor()
            self.logger.info("✅ Preprocessor initialized")
            
            # Initialize AI models dengan dummy shape (akan di-update saat training)
            self.ai_models = SensorAIModels(input_shape=(60, 30))
            self.logger.info("✅ AI models initialized")
            
            # Setup automatic tasks
            self._setup_scheduled_tasks()
            
            self.logger.info("🎉 AI System Manager berhasil diinisialisasi!")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing system: {e}")
            raise
    
    def _setup_scheduled_tasks(self):
        """Setup automatic scheduled tasks"""
        # Auto retrain setiap 6 jam
        schedule.every(6).hours.do(self._scheduled_retrain)
        
        # Health check setiap 1 jam
        schedule.every(1).hours.do(self._scheduled_health_check)
        
        # System cleanup setiap 24 jam
        schedule.every(24).hours.do(self._scheduled_cleanup)
        
        self.logger.info("📅 Scheduled tasks berhasil di-setup")
    
    def train_initial_models(self, data_limit=5000):
        """Training awal untuk semua models"""
        try:
            self.logger.info(f"🎯 Memulai initial training dengan {data_limit} data...")
            
            # Load data from database
            df = self._load_training_data(data_limit)
            if df is None or df.empty:
                self.logger.error("❌ Tidak dapat memuat data untuk training")
                return False
            
            # Prepare data
            X, y = self._prepare_training_data(df)
            if X is None or y is None:
                self.logger.error("❌ Gagal menyiapkan data training")
                return False
            
            # Update input shape
            self.ai_models.input_shape = (X.shape[1], X.shape[2])
            
            # Build models
            self.logger.info("🏗️ Building AI models...")
            self.ai_models.build_cnn_model()
            self.ai_models.build_lstm_model()
            self.ai_models.build_hybrid_cnn_lstm_model()
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train each model
            for model_type in ['CNN', 'LSTM', 'CNN_LSTM']:
                self.logger.info(f"🚀 Training {model_type} model...")
                try:
                    self.ai_models.train_model(
                        model_type, X_train, y_train, X_val, y_val,
                        epochs=50, batch_size=32, verbose=1
                    )
                    self.logger.info(f"✅ {model_type} training selesai")
                except Exception as e:
                    self.logger.error(f"❌ Error training {model_type}: {e}")
            
            # Save models
            saved_models = self.ai_models.save_models()
            self.last_training = datetime.now()
            
            self.logger.info(f"🎉 Initial training selesai! Models saved: {saved_models}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam initial training: {e}")
            return False
    
    def _load_training_data(self, limit):
        """Load data untuk training dari database"""
        try:
            query = f"""
            SELECT timestamp, ph, suhu, kualitas 
            FROM sensor_readings 
            ORDER BY timestamp DESC 
            LIMIT {limit}
            """
            
            df = self.db_manager.fetch_data(query)
            if df.empty:
                self.logger.warning("⚠️ Tidak ada data dari database")
                return None
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.logger.info(f"📊 Loaded {len(df)} records for training")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading training data: {e}")
            return None
    
    def _prepare_training_data(self, df):
        """Prepare data untuk training"""
        try:
            # Feature engineering
            df_features = self.preprocessor.create_features(df)
            
            # Fit scalers
            self.preprocessor.fit_scalers(df_features)
            
            # Transform features
            df_transformed = self.preprocessor.transform_features(df_features)
            
            # Prepare sequences
            X, y = self.preprocessor.prepare_sequence_data(df_transformed)
            
            if X is not None and y is not None:
                self.logger.info(f"✅ Training data prepared: X={X.shape}, y={y.shape}")
                return X, y
            else:
                return None, None
                
        except Exception as e:
            self.logger.error(f"❌ Error preparing training data: {e}")
            return None, None
    
    def predict_realtime_batch(self, limit=100):
        """Prediksi batch untuk data terbaru"""
        try:
            # Get latest data
            df = self._load_training_data(limit)
            if df is None or df.empty:
                return None
            
            # Get predictions untuk setiap row
            predictions = []
            for idx, row in df.iterrows():
                latest_window = df.iloc[max(0, idx-59):idx+1]  # 60 timesteps window
                
                if len(latest_window) >= 10:  # Minimum data required
                    pred = self.ai_models.predict_realtime(latest_window, return_probabilities=True)
                    if pred:
                        pred['actual_ph'] = row['ph']
                        pred['actual_suhu'] = row['suhu'] 
                        pred['actual_kualitas'] = row['kualitas']
                        pred['timestamp_actual'] = row['timestamp']
                        predictions.append(pred)
            
            self.prediction_count += len(predictions)
            self.logger.info(f"📊 Batch prediction completed: {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam batch prediction: {e}")
            return None
    
    def _scheduled_retrain(self):
        """Automatic retraining yang dijadwalkan"""
        if not self.auto_retrain_enabled:
            return
        
        self.logger.info("🔄 Starting scheduled retrain...")
        success = self.ai_models.auto_retrain(min_new_data=200)
        
        if success:
            self.last_training = datetime.now()
            self.logger.info("✅ Scheduled retrain completed")
        else:
            self.logger.warning("⚠️ Scheduled retrain failed or skipped")
    
    def _scheduled_health_check(self):
        """Health check yang dijadwalkan"""
        try:
            health = self.ai_models.health_check()
            self.logger.info(f"💚 Health check: {health}")
            
            # Alert jika ada masalah
            if health['models_loaded'] == 0:
                self.logger.warning("⚠️ ALERT: Tidak ada model yang loaded!")
            
            if not health['database_connected']:
                self.logger.warning("⚠️ ALERT: Database tidak terkoneksi!")
                
        except Exception as e:
            self.logger.error(f"❌ Error dalam health check: {e}")
    
    def _scheduled_cleanup(self):
        """Cleanup system resources"""
        try:
            # Clear prediction buffer yang terlalu lama
            if self.ai_models.prediction_buffer:
                cutoff = datetime.now() - timedelta(hours=24)
                self.ai_models.prediction_buffer = [
                    p for p in self.ai_models.prediction_buffer 
                    if p['timestamp'] > cutoff
                ]
            
            # Clear system errors yang lama
            if self.system_errors:
                cutoff = datetime.now() - timedelta(hours=24)
                self.system_errors = [
                    e for e in self.system_errors 
                    if e.get('timestamp', datetime.now()) > cutoff
                ]
            
            self.logger.info("🧹 System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam cleanup: {e}")
    
    def start_background_tasks(self):
        """Start background tasks dalam thread terpisah"""
        if self.is_running:
            self.logger.warning("⚠️ Background tasks sudah berjalan")
            return
        
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        self.logger.info("🚀 Background tasks started")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.is_running = False
        self.logger.info("⏹️ Background tasks stopped")
    
    def get_system_status(self):
        """Dapatkan status lengkap sistem"""
        try:
            status = {
                'timestamp': datetime.now(),
                'system_running': self.is_running,
                'last_training': self.last_training,
                'prediction_count': self.prediction_count,
                'auto_retrain_enabled': self.auto_retrain_enabled,
                'components': {
                    'database': self.db_manager is not None,
                    'preprocessor': self.preprocessor is not None,
                    'ai_models': self.ai_models is not None
                }
            }
            
            # AI models health check
            if self.ai_models:
                ai_health = self.ai_models.health_check()
                status['ai_health'] = ai_health
            
            # Recent prediction stats
            if self.ai_models:
                recent_stats = self.ai_models.get_prediction_stats(hours=1)
                status['recent_predictions'] = recent_stats
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system status: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Initialize dan test system
    print("🚀 Starting AI System Manager...")
    
    manager = AISystemManager()
    
    # Check if models exist, if not do initial training
    try:
        manager.ai_models.load_models()
        if not any(manager.ai_models.is_trained.values()):
            print("📚 No trained models found, starting initial training...")
            success = manager.train_initial_models(data_limit=2000)
            if not success:
                print("❌ Initial training failed!")
                exit(1)
        else:
            print("✅ Pre-trained models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("📚 Starting fresh training...")
        success = manager.train_initial_models(data_limit=2000)
        if not success:
            print("❌ Training failed!")
            exit(1)
    
    # Start background tasks
    manager.start_background_tasks()
    
    # Test predictions
    print("\n🔮 Testing real-time predictions...")
    predictions = manager.predict_realtime_batch(limit=50)
    if predictions:
        print(f"✅ Generated {len(predictions)} predictions")
        
        # Show sample predictions
        for i, pred in enumerate(predictions[:3]):
            print(f"   Prediction {i+1}:")
            print(f"     pH: {pred['ph_predicted']:.2f} (actual: {pred['actual_ph']:.2f})")
            print(f"     Suhu: {pred['suhu_predicted']:.2f} (actual: {pred['actual_suhu']:.2f})")
            print(f"     Kualitas: {pred['kualitas_predicted']} (actual: {pred['actual_kualitas']})")
            print(f"     Confidence: {pred['confidence']:.3f}")
    
    # Show system status
    print("\n📊 System Status:")
    status = manager.get_system_status()
    for key, value in status.items():
        if key != 'ai_health' and key != 'recent_predictions':
            print(f"   {key}: {value}")
    
    print("\n🎉 AI System Manager is running!")
    print("💡 Tekan Ctrl+C untuk stop...")
    
    try:
        while True:
            time.sleep(30)
            
            # Show periodic updates
            status = manager.get_system_status()
            if status.get('recent_predictions'):
                recent = status['recent_predictions']
                print(f"📈 Last hour: {recent['count']} predictions, avg confidence: {recent['avg_confidence']:.3f}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping AI System Manager...")
        manager.stop_background_tasks()
        print("✅ System stopped successfully")
