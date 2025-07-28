#!/usr/bin/env python3
"""
Real-time Data Preprocessing untuk Sensor Data
Feature Engineering & Data Preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class SensorDataPreprocessor:
    """Real-time preprocessing untuk data sensor"""
    
    def __init__(self):
        self.scaler_features = StandardScaler()
        self.scaler_targets = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Parameter untuk sliding window
        self.window_size = 50  # 50 detik data untuk prediksi
        self.prediction_horizon = 10  # prediksi 10 detik ke depan
        
    def create_features(self, df):
        """Buat fitur engineering dari data sensor"""
        if df.empty or len(df) < 10:
            return df
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Moving averages
        df['ph_ma_5'] = df['ph'].rolling(window=5, min_periods=1).mean()
        df['ph_ma_10'] = df['ph'].rolling(window=10, min_periods=1).mean()
        df['suhu_ma_5'] = df['suhu'].rolling(window=5, min_periods=1).mean()
        df['suhu_ma_10'] = df['suhu'].rolling(window=10, min_periods=1).mean()
        
        # 2. Standard deviation (volatility)
        df['ph_std_5'] = df['ph'].rolling(window=5, min_periods=1).std().fillna(0)
        df['suhu_std_5'] = df['suhu'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # 3. Rate of change
        df['ph_change'] = df['ph'].diff().fillna(0)
        df['suhu_change'] = df['suhu'].diff().fillna(0)
        df['ph_change_rate'] = df['ph'].pct_change().fillna(0)
        df['suhu_change_rate'] = df['suhu'].pct_change().fillna(0)
        
        # 4. Min/Max dalam window
        df['ph_min_10'] = df['ph'].rolling(window=10, min_periods=1).min()
        df['ph_max_10'] = df['ph'].rolling(window=10, min_periods=1).max()
        df['suhu_min_10'] = df['suhu'].rolling(window=10, min_periods=1).min()
        df['suhu_max_10'] = df['suhu'].rolling(window=10, min_periods=1).max()
        
        # 5. Range features
        df['ph_range_10'] = df['ph_max_10'] - df['ph_min_10']
        df['suhu_range_10'] = df['suhu_max_10'] - df['suhu_min_10']
        
        # 6. Lag features
        for lag in [1, 2, 3, 5]:
            df[f'ph_lag_{lag}'] = df['ph'].shift(lag).fillna(df['ph'].iloc[0])
            df[f'suhu_lag_{lag}'] = df['suhu'].shift(lag).fillna(df['suhu'].iloc[0])
        
        # 7. Exponential weighted moving average
        df['ph_ewm'] = df['ph'].ewm(span=10).mean()
        df['suhu_ewm'] = df['suhu'].ewm(span=10).mean()
        
        # 8. Interaction features
        df['ph_suhu_interaction'] = df['ph'] * df['suhu']
        df['ph_suhu_ratio'] = df['ph'] / (df['suhu'] + 1e-6)
        
        # 9. Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['second'] = df['timestamp'].dt.second
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 10. Quality-based statistical features
        df['quality_encoded'] = self.label_encoder.fit_transform(df['kualitas']) if not self.is_fitted else self.label_encoder.transform(df['kualitas'])
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def prepare_sequence_data(self, df, target_cols=['ph', 'suhu', 'kualitas']):
        """Siapkan data sequence untuk CNN/LSTM"""
        if len(df) < self.window_size + self.prediction_horizon:
            return None, None
        
        # Fitur untuk model
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'no', 'id'] + target_cols]
        
        X_sequences = []
        y_sequences = []
        
        # Buat sequences
        for i in range(len(df) - self.window_size - self.prediction_horizon + 1):
            # Input sequence (window_size langkah ke belakang)
            X_seq = df[feature_cols].iloc[i:i + self.window_size].values
            
            # Target sequence (prediction_horizon langkah ke depan)
            target_idx = i + self.window_size + self.prediction_horizon - 1
            if target_idx < len(df):
                y_seq = [
                    df['ph'].iloc[target_idx],
                    df['suhu'].iloc[target_idx],
                    1 if df['kualitas'].iloc[target_idx] == 'baik' else 0
                ]
                
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
        
        if not X_sequences:
            return None, None
            
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        return X, y
    
    def fit_scalers(self, df):
        """Fit scalers dengan data training"""
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'no', 'id', 'kualitas']]
        
        # Fit feature scaler
        self.scaler_features.fit(df[feature_cols])
        
        # Fit target scaler untuk regression targets
        target_regression = df[['ph', 'suhu']]
        self.scaler_targets.fit(target_regression)
        
        # Fit label encoder
        self.label_encoder.fit(df['kualitas'])
        
        self.is_fitted = True
        print("✅ Scalers berhasil di-fit dengan data training")
    
    def transform_features(self, df):
        """Transform features menggunakan fitted scalers"""
        if not self.is_fitted:
            raise ValueError("Scalers belum di-fit! Jalankan fit_scalers() terlebih dahulu.")
        
        df_transformed = df.copy()
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'no', 'id', 'kualitas']]
        
        # Transform features
        df_transformed[feature_cols] = self.scaler_features.transform(df[feature_cols])
        
        return df_transformed
    
    def inverse_transform_targets(self, y_pred):
        """Inverse transform untuk target predictions"""
        # y_pred format: [ph, suhu, kualitas_prob]
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(1, -1)
        
        # Inverse transform untuk ph dan suhu
        regression_preds = self.scaler_targets.inverse_transform(y_pred[:, :2])
        
        # Classification prediction tetap sama
        classification_preds = y_pred[:, 2:] if y_pred.shape[1] > 2 else y_pred[:, 2].reshape(-1, 1)
        
        return regression_preds, classification_preds
    
    def prepare_realtime_input(self, latest_data):
        """Siapkan input untuk prediksi real-time"""
        if len(latest_data) < self.window_size:
            # Jika data kurang, pad dengan data terakhir
            last_row = latest_data.iloc[-1:].copy()
            padding_needed = self.window_size - len(latest_data)
            
            padding_data = pd.concat([last_row] * padding_needed, ignore_index=True)
            latest_data = pd.concat([padding_data, latest_data], ignore_index=True)
        
        # Ambil window terakhir
        window_data = latest_data.tail(self.window_size).copy()
        
        # Create features
        window_data = self.create_features(window_data)
        
        # Transform
        window_data = self.transform_features(window_data)
        
        # Siapkan untuk model
        feature_cols = [col for col in window_data.columns if col not in ['timestamp', 'no', 'id', 'kualitas']]
        X = window_data[feature_cols].values.reshape(1, self.window_size, -1)
        
        return X
    
    def get_feature_names(self, df_sample):
        """Dapatkan nama-nama fitur yang akan digunakan"""
        df_with_features = self.create_features(df_sample)
        feature_cols = [col for col in df_with_features.columns if col not in ['timestamp', 'no', 'id', 'kualitas']]
        return feature_cols
    
    def validate_data_quality(self, df):
        """Validasi kualitas data"""
        issues = []
        
        # Check missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        critical_missing = missing_pct[missing_pct > 50]
        if not critical_missing.empty:
            issues.append(f"Kolom dengan missing values >50%: {list(critical_missing.index)}")
        
        # Check outliers (simple IQR method)
        for col in ['ph', 'suhu']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > len(df) * 0.1:  # >10% outliers
                    issues.append(f"Kolom {col} memiliki {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
        
        # Check data consistency
        if 'ph' in df.columns:
            invalid_ph = df[(df['ph'] < 0) | (df['ph'] > 14)]
            if not invalid_ph.empty:
                issues.append(f"Nilai pH tidak valid: {len(invalid_ph)} baris")
        
        if 'suhu' in df.columns:
            invalid_suhu = df[(df['suhu'] < -50) | (df['suhu'] > 100)]
            if not invalid_suhu.empty:
                issues.append(f"Nilai suhu tidak valid: {len(invalid_suhu)} baris")
        
        return issues

if __name__ == "__main__":
    # Test preprocessing
    print("🧪 Testing Sensor Data Preprocessor...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'no': range(1, n_samples + 1),
        'ph': np.random.uniform(4, 10, n_samples),
        'suhu': np.random.uniform(15, 40, n_samples),
        'kualitas': np.random.choice(['baik', 'buruk'], n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1S')
    })
    
    # Test preprocessor
    preprocessor = SensorDataPreprocessor()
    
    print("📊 Data asli:", sample_data.shape)
    
    # Create features
    featured_data = preprocessor.create_features(sample_data)
    print("🔧 Setelah feature engineering:", featured_data.shape)
    print("📋 Fitur yang dibuat:", [col for col in featured_data.columns if col not in sample_data.columns])
    
    # Fit scalers
    preprocessor.fit_scalers(featured_data)
    
    # Transform
    transformed_data = preprocessor.transform_features(featured_data)
    print("⚡ Data berhasil di-transform")
    
    # Prepare sequences
    X, y = preprocessor.prepare_sequence_data(transformed_data)
    if X is not None:
        print(f"🔄 Sequence data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Validate data quality
    issues = preprocessor.validate_data_quality(sample_data)
    if issues:
        print("⚠️ Data quality issues:", issues)
    else:
        print("✅ Data quality: OK")
    
    print("✅ Preprocessing test selesai!")
