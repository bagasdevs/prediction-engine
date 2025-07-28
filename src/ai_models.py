#!/usr/bin/env python3
"""
AI Models untuk Real-time Sensor Data Prediction
CNN, LSTM, dan Hybrid CNN-LSTM Models
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential  # type: ignore
    from tensorflow.keras.layers import (  # type: ignore
        Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, 
        Input, Concatenate, BatchNormalization, 
        MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorFlow tidak tersedia: {e}")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes untuk type hints
    Model = None
    Sequential = None
    Dense = None
    LSTM = None
    Conv1D = None
    Adam = None

# Sklearn imports
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Scikit-learn tidak tersedia")
    SKLEARN_AVAILABLE = False
    # Create dummy functions
    accuracy_score = lambda x, y: 0.0
    precision_score = lambda x, y, **kwargs: 0.0
    recall_score = lambda x, y, **kwargs: 0.0
    f1_score = lambda x, y, **kwargs: 0.0
    mean_squared_error = lambda x, y: 0.0

# Real-time system imports
import mysql.connector
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Tuple, Optional, Any
import logging

# Conditional imports for internal modules
try:
    from .realtime_preprocessor import SensorDataPreprocessor
    from .database_manager import DatabaseManager
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from realtime_preprocessor import SensorDataPreprocessor
        from database_manager import DatabaseManager
    except ImportError as e:
        print(f"⚠️ Warning: {e}")
        SensorDataPreprocessor = None
        DatabaseManager = None

class SensorAIModels:
    """AI Models untuk prediksi sensor data real-time"""
    
    def __init__(self, input_shape, output_dim=3):
        self.input_shape = input_shape  # (timesteps, features)
        self.output_dim = output_dim    # [ph, suhu, kualitas]
        self.models = {}
        self.model_history = {}
        self.is_trained = {}
        
        # Check TensorFlow availability
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow tidak tersedia. Install dengan: pip install tensorflow")
        
        # Real-time system components
        self.db_manager = None
        self.preprocessor = None
        self.logger = self._setup_logging()
        self.model_performance = {}
        self.prediction_buffer = []
        
        # Configuration
        self.sequence_length = 60
        self.prediction_threshold = 0.5
        self.max_buffer_size = 1000
        
        # Setup GPU dan initialize components
        self.setup_gpu()
        self._initialize_components()
    
    def _setup_logging(self):
        """Setup logging untuk monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_models.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize database manager dan preprocessor"""
        try:
            if DatabaseManager is not None:
                self.db_manager = DatabaseManager()
            if SensorDataPreprocessor is not None:
                self.preprocessor = SensorDataPreprocessor()
            self.logger.info("✅ Database manager dan preprocessor berhasil diinisialisasi")
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            # Fallback mode tanpa database
            self.db_manager = None
            self.logger.warning("⚠️ Menjalankan dalam mode offline")
    
    def setup_gpu(self):
        """Setup GPU untuk training yang lebih cepat"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU tersedia: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(f"⚠️ GPU setup error: {e}")
        else:
            print("💻 Menggunakan CPU untuk training")
    
    def build_cnn_model(self, filters=[64, 128, 64], kernel_sizes=[3, 3, 3], pool_sizes=[2, 2, 2]):
        """Build CNN model untuk time series classification & regression"""
        
        model = Sequential([
            # Input layer
            Input(shape=self.input_shape),
            
            # CNN layers
            Conv1D(filters=filters[0], kernel_size=kernel_sizes[0], activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=pool_sizes[0]),
            Dropout(0.2),
            
            Conv1D(filters=filters[1], kernel_size=kernel_sizes[1], activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=pool_sizes[1]),
            Dropout(0.3),
            
            Conv1D(filters=filters[2], kernel_size=kernel_sizes[2], activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.4),
            
            # Dense layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2)
        ])
        
        # Output layers - Multi-task learning
        # Regression outputs (ph, suhu)
        regression_output = Dense(2, activation='linear', name='regression_output')(model.output)
        
        # Classification output (kualitas)
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(model.output)
        
        # Combine outputs
        final_model = Model(inputs=model.input, 
                          outputs=[regression_output, classification_output])
        
        final_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression_output': 'mse',
                'classification_output': 'binary_crossentropy'
            },
            loss_weights={
                'regression_output': 1.0,
                'classification_output': 2.0  # Lebih fokus pada klasifikasi
            },
            metrics={
                'regression_output': ['mae'],
                'classification_output': ['accuracy']
            }
        )
        
        self.models['CNN'] = final_model
        print("✅ CNN Model berhasil dibuat")
        print(f"📊 Parameters: {final_model.count_params():,}")
        
        return final_model
    
    def build_lstm_model(self, lstm_units=[128, 64, 32], return_sequences=[True, True, False]):
        """Build LSTM model untuk time series prediction"""
        
        model = Sequential([
            Input(shape=self.input_shape),
            
            # LSTM layers
            LSTM(lstm_units[0], return_sequences=return_sequences[0], dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            LSTM(lstm_units[1], return_sequences=return_sequences[1], dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            LSTM(lstm_units[2], return_sequences=return_sequences[2], dropout=0.4, recurrent_dropout=0.4),
            BatchNormalization(),
            
            # Dense layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            Dropout(0.3)
        ])
        
        # Multi-task outputs
        regression_output = Dense(2, activation='linear', name='regression_output')(model.output)
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(model.output)
        
        final_model = Model(inputs=model.input, 
                          outputs=[regression_output, classification_output])
        
        final_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression_output': 'mse',
                'classification_output': 'binary_crossentropy'
            },
            loss_weights={
                'regression_output': 1.0,
                'classification_output': 2.0
            },
            metrics={
                'regression_output': ['mae'],
                'classification_output': ['accuracy']
            }
        )
        
        self.models['LSTM'] = final_model
        print("✅ LSTM Model berhasil dibuat")
        print(f"📊 Parameters: {final_model.count_params():,}")
        
        return final_model
    
    def build_hybrid_cnn_lstm_model(self):
        """Build Hybrid CNN-LSTM model"""
        
        # Input
        input_layer = Input(shape=self.input_shape)
        
        # CNN branch
        cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Dropout(0.2)(cnn_branch)
        
        cnn_branch = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(cnn_branch)
        cnn_branch = BatchNormalization()(cnn_branch)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Dropout(0.3)(cnn_branch)
        
        # LSTM branch (menggunakan output CNN)
        lstm_branch = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(cnn_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        lstm_branch = LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)(lstm_branch)
        lstm_branch = BatchNormalization()(lstm_branch)
        
        # Attention mechanism
        attention_layer = Dense(64, activation='tanh')(lstm_branch)
        attention_weights = Dense(1, activation='softmax')(attention_layer)
        attended_features = tf.multiply(lstm_branch, attention_weights)
        
        # Dense layers
        dense_layer = Dense(256, activation='relu')(attended_features)
        dense_layer = BatchNormalization()(dense_layer)
        dense_layer = Dropout(0.5)(dense_layer)
        
        dense_layer = Dense(128, activation='relu')(dense_layer)
        dense_layer = BatchNormalization()(dense_layer)
        dense_layer = Dropout(0.4)(dense_layer)
        
        dense_layer = Dense(64, activation='relu')(dense_layer)
        dense_layer = Dropout(0.3)(dense_layer)
        
        # Multi-task outputs
        regression_output = Dense(2, activation='linear', name='regression_output')(dense_layer)
        classification_output = Dense(1, activation='sigmoid', name='classification_output')(dense_layer)
        
        final_model = Model(inputs=input_layer, 
                          outputs=[regression_output, classification_output])
        
        final_model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Learning rate lebih kecil untuk model kompleks
            loss={
                'regression_output': 'mse',
                'classification_output': 'binary_crossentropy'
            },
            loss_weights={
                'regression_output': 1.0,
                'classification_output': 2.5  # Weight lebih besar untuk klasifikasi
            },
            metrics={
                'regression_output': ['mae'],
                'classification_output': ['accuracy']
            }
        )
        
        self.models['CNN_LSTM'] = final_model
        print("✅ Hybrid CNN-LSTM Model berhasil dibuat")
        print(f"📊 Parameters: {final_model.count_params():,}")
        
        return final_model
    
    def train_model(self, model_type, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, verbose=1):
        """Train model dengan early stopping dan callbacks"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} belum dibuat!")
        
        model = self.models[model_type]
        
        # Prepare targets untuk multi-task learning
        y_train_reg = y_train[:, :2]  # ph, suhu
        y_train_clf = y_train[:, 2:3]  # kualitas
        
        y_val_reg = y_val[:, :2]
        y_val_clf = y_val[:, 2:3]
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'models/best_{model_type.lower()}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"🚀 Memulai training {model_type} model...")
        print(f"📊 Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        # Training
        history = model.fit(
            X_train,
            {
                'regression_output': y_train_reg,
                'classification_output': y_train_clf
            },
            validation_data=(
                X_val,
                {
                    'regression_output': y_val_reg,
                    'classification_output': y_val_clf
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.model_history[model_type] = history
        self.is_trained[model_type] = True
        
        print(f"✅ Training {model_type} selesai!")
        
        return history
    
    def predict(self, model_type, X, return_proba=False):
        """Prediksi menggunakan model yang dipilih"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} belum dibuat!")
        
        if not self.is_trained.get(model_type, False):
            raise ValueError(f"Model {model_type} belum di-train!")
        
        model = self.models[model_type]
        predictions = model.predict(X, verbose=0)
        
        # predictions = [regression_output, classification_output]
        reg_pred = predictions[0]  # ph, suhu
        clf_pred = predictions[1]  # kualitas probability
        
        if return_proba:
            return reg_pred, clf_pred
        else:
            # Convert classification probability to binary
            clf_binary = (clf_pred > 0.5).astype(int)
            return reg_pred, clf_binary
    
    def evaluate_model(self, model_type, X_test, y_test):
        """Evaluasi performa model"""
        
        if model_type not in self.models or not self.is_trained.get(model_type, False):
            raise ValueError(f"Model {model_type} belum siap untuk evaluasi!")
        
        # Prediksi
        reg_pred, clf_pred = self.predict(model_type, X_test, return_proba=True)
        
        # Regression metrics (ph, suhu)
        y_reg_true = y_test[:, :2]
        rmse_ph = np.sqrt(mean_squared_error(y_reg_true[:, 0], reg_pred[:, 0]))
        rmse_suhu = np.sqrt(mean_squared_error(y_reg_true[:, 1], reg_pred[:, 1]))
        
        # Classification metrics (kualitas)
        y_clf_true = y_test[:, 2]
        clf_binary = (clf_pred.flatten() > 0.5).astype(int)
        
        accuracy = accuracy_score(y_clf_true, clf_binary)
        precision = precision_score(y_clf_true, clf_binary, zero_division=0)
        recall = recall_score(y_clf_true, clf_binary, zero_division=0)
        f1 = f1_score(y_clf_true, clf_binary, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'rmse_ph': rmse_ph,
            'rmse_suhu': rmse_suhu,
            'training_samples': len(X_test)
        }
        
        print(f"📊 Evaluasi {model_type}:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   RMSE pH: {rmse_ph:.4f}")
        print(f"   RMSE Suhu: {rmse_suhu:.4f}")
        
        return metrics
    
    def create_ensemble_prediction(self, X):
        """Buat prediksi ensemble dari semua model"""
        
        available_models = [model_type for model_type in ['CNN', 'LSTM', 'CNN_LSTM'] 
                          if self.is_trained.get(model_type, False)]
        
        if not available_models:
            raise ValueError("Tidak ada model yang sudah di-train!")
        
        reg_predictions = []
        clf_predictions = []
        
        for model_type in available_models:
            reg_pred, clf_pred = self.predict(model_type, X, return_proba=True)
            reg_predictions.append(reg_pred)
            clf_predictions.append(clf_pred)
        
        # Ensemble averaging
        ensemble_reg = np.mean(reg_predictions, axis=0)
        ensemble_clf = np.mean(clf_predictions, axis=0)
        
        return ensemble_reg, ensemble_clf
    
    def save_models(self, save_dir='models'):
        """Simpan semua model yang sudah di-train"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_models = []
        for model_type, model in self.models.items():
            if self.is_trained.get(model_type, False):
                model_path = f"{save_dir}/{model_type.lower()}_sensor_model.h5"
                model.save(model_path)
                saved_models.append(model_type)
                print(f"✅ {model_type} model disimpan: {model_path}")
        
        # Simpan metadata
        metadata = {
            'input_shape': self.input_shape,
            'output_dim': self.output_dim,
            'trained_models': saved_models
        }
        
        metadata_path = f"{save_dir}/model_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        print(f"📋 Metadata disimpan: {metadata_path}")
        
        return saved_models
    
    def load_models(self, save_dir='models'):
        """Load model yang sudah disimpan"""
        import os
        
        # Load metadata
        metadata_path = f"{save_dir}/model_metadata.joblib"
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            print(f"📋 Metadata dimuat: {metadata}")
        else:
            print("⚠️ Metadata tidak ditemukan")
            return
        
        # Load models
        for model_type in metadata['trained_models']:
            model_path = f"{save_dir}/{model_type.lower()}_sensor_model.h5"
            if os.path.exists(model_path):
                self.models[model_type] = tf.keras.models.load_model(model_path)
                self.is_trained[model_type] = True
                print(f"✅ {model_type} model dimuat: {model_path}")
        
        print(f"📊 Total model dimuat: {len([m for m in self.is_trained.values() if m])}")
    
    def load_realtime_data(self, limit=1000):
        """Load data terbaru dari database untuk training/evaluation"""
        if not self.db_manager:
            self.logger.error("❌ Database manager tidak tersedia")
            return None
        
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
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.logger.info(f"📊 Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data from database: {e}")
            return None
    
    def prepare_realtime_training_data(self, df):
        """Siapkan data untuk training dari database"""
        if df is None or df.empty:
            return None, None
        
        try:
            # Feature engineering
            df_features = self.preprocessor.create_features(df)
            
            # Fit scalers jika belum
            if not self.preprocessor.is_fitted:
                self.preprocessor.fit_scalers(df_features)
            
            # Transform features
            df_transformed = self.preprocessor.transform_features(df_features)
            
            # Prepare sequences
            X, y = self.preprocessor.prepare_sequence_data(df_transformed)
            
            if X is not None and y is not None:
                self.logger.info(f"✅ Data siap untuk training: X={X.shape}, y={y.shape}")
                return X, y
            else:
                self.logger.warning("⚠️ Gagal menyiapkan sequence data")
                return None, None
                
        except Exception as e:
            self.logger.error(f"❌ Error preparing training data: {e}")
            return None, None
    
    def predict_realtime(self, latest_data, return_probabilities=False):
        """Prediksi real-time dari data terbaru"""
        if not any(self.is_trained.values()):
            self.logger.error("❌ Tidak ada model yang sudah di-train!")
            return None
        
        try:
            # Prepare input data
            input_data = self.preprocessor.prepare_realtime_input(latest_data)
            if input_data is None:
                self.logger.warning("⚠️ Gagal menyiapkan input data")
                return None
            
            # Get ensemble prediction
            reg_pred, clf_pred = self.create_ensemble_prediction(input_data)
            
            # Process results
            ph_pred = float(reg_pred[0, 0])
            suhu_pred = float(reg_pred[0, 1])
            kualitas_prob = float(clf_pred[0, 0])
            kualitas_pred = int(kualitas_prob > self.prediction_threshold)
            
            # Store in buffer for monitoring
            prediction_result = {
                'timestamp': datetime.now(),
                'ph_predicted': ph_pred,
                'suhu_predicted': suhu_pred,
                'kualitas_probability': kualitas_prob,
                'kualitas_predicted': kualitas_pred,
                'confidence': kualitas_prob if kualitas_pred == 1 else (1 - kualitas_prob)
            }
            
            self._add_to_buffer(prediction_result)
            
            if return_probabilities:
                return prediction_result
            else:
                return {
                    'ph': ph_pred,
                    'suhu': suhu_pred, 
                    'kualitas': kualitas_pred
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error dalam prediksi real-time: {e}")
            return None
    
    def _add_to_buffer(self, prediction):
        """Tambahkan prediksi ke buffer untuk monitoring"""
        self.prediction_buffer.append(prediction)
        
        # Maintain buffer size
        if len(self.prediction_buffer) > self.max_buffer_size:
            self.prediction_buffer = self.prediction_buffer[-self.max_buffer_size:]
    
    def get_prediction_stats(self, hours=1):
        """Dapatkan statistik prediksi dalam periode tertentu"""
        if not self.prediction_buffer:
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = [
            p for p in self.prediction_buffer 
            if p['timestamp'] > cutoff_time
        ]
        
        if not recent_predictions:
            return None
        
        stats = {
            'count': len(recent_predictions),
            'avg_ph': np.mean([p['ph_predicted'] for p in recent_predictions]),
            'avg_suhu': np.mean([p['suhu_predicted'] for p in recent_predictions]),
            'avg_confidence': np.mean([p['confidence'] for p in recent_predictions]),
            'quality_good_ratio': np.mean([p['kualitas_predicted'] for p in recent_predictions]),
            'period_hours': hours
        }
        
        return stats
    
    def auto_retrain(self, min_new_data=100):
        """Automatic retraining jika ada data baru yang cukup"""
        if not self.db_manager:
            self.logger.warning("⚠️ Auto retrain tidak tersedia tanpa database")
            return False
        
        try:
            # Check untuk data baru
            latest_data = self.load_realtime_data(limit=min_new_data * 2)
            if latest_data is None or len(latest_data) < min_new_data:
                self.logger.info(f"📊 Data tidak cukup untuk retrain ({len(latest_data) if latest_data is not None else 0} < {min_new_data})")
                return False
            
            # Prepare data
            X, y = self.prepare_realtime_training_data(latest_data)
            if X is None or y is None:
                self.logger.warning("⚠️ Gagal menyiapkan data untuk retrain")
                return False
            
            # Split for training and validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.logger.info(f"🚀 Memulai auto retrain dengan {len(X_train)} training samples")
            
            # Retrain models yang sudah ada
            retrained_models = []
            for model_type in self.models.keys():
                if self.is_trained.get(model_type, False):
                    try:
                        self.train_model(
                            model_type, X_train, y_train, X_val, y_val,
                            epochs=20, batch_size=32, verbose=0
                        )
                        retrained_models.append(model_type)
                        self.logger.info(f"✅ {model_type} berhasil di-retrain")
                    except Exception as e:
                        self.logger.error(f"❌ Error retraining {model_type}: {e}")
            
            if retrained_models:
                # Auto save models
                self.save_models()
                self.logger.info(f"✅ Auto retrain selesai: {len(retrained_models)} models")
                return True
            else:
                self.logger.warning("⚠️ Tidak ada model yang berhasil di-retrain")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error dalam auto retrain: {e}")
            return False
    
    def health_check(self):
        """Check kesehatan sistem AI"""
        health_status = {
            'timestamp': datetime.now(),
            'models_loaded': len([m for m in self.is_trained.values() if m]),
            'total_models': len(self.models),
            'database_connected': self.db_manager is not None,
            'preprocessor_fitted': self.preprocessor.is_fitted if self.preprocessor else False,
            'prediction_buffer_size': len(self.prediction_buffer),
            'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0
        }
        
        # Recent performance
        recent_stats = self.get_prediction_stats(hours=1)
        if recent_stats:
            health_status['recent_predictions'] = recent_stats['count']
            health_status['avg_confidence'] = recent_stats['avg_confidence']
        else:
            health_status['recent_predictions'] = 0
            health_status['avg_confidence'] = 0.0
        
        return health_status

if __name__ == "__main__":
    # Test AI Models
    print("🤖 Testing AI Models...")
    
    # Simulate data
    np.random.seed(42)
    timesteps, features = 50, 30
    samples = 1000
    
    X = np.random.randn(samples, timesteps, features)
    y = np.random.rand(samples, 3)  # [ph, suhu, kualitas]
    y[:, 2] = (y[:, 2] > 0.5).astype(int)  # Binary untuk kualitas
    
    print(f"📊 Data shape: X={X.shape}, y={y.shape}")
    
    # Initialize models
    ai_models = SensorAIModels(input_shape=(timesteps, features))
    
    # Build models
    ai_models.build_cnn_model()
    ai_models.build_lstm_model()
    ai_models.build_hybrid_cnn_lstm_model()
    
    print("✅ Semua model berhasil dibuat!")
    print(f"📋 Available models: {list(ai_models.models.keys())}")
