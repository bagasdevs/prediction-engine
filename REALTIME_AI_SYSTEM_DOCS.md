# Real-time Sensor AI System - Dokumentasi Lengkap

## 🎯 Overview Sistem

Sistem AI Real-time untuk prediksi sensor data dengan menggunakan:
- **CNN** (Convolutional Neural Network)
- **LSTM** (Long Short-Term Memory) 
- **Hybrid CNN-LSTM**
- **Ensemble Methods** dengan robustness testing

## 🏗️ Arsitektur Sistem

```
Database (MySQL) → Data Processing → AI Models → Real-time Dashboard
       ↑                ↑              ↑           ↑
   simulasi.py   realtime_preprocessor  ai_models   realtime_dashboard
                                             ↓
                         robustness_test.py (Testing)
```

## 📁 Struktur File

```
📦 Real-time Sensor AI System
├── 🗃️ database_manager.py      # Database operations & health monitoring
├── ⚙️ realtime_preprocessor.py # Feature engineering & data preparation  
├── 🤖 ai_models.py             # CNN, LSTM, Hybrid models
├── 🔄 realtime_processor.py    # Main processing engine
├── 📊 realtime_dashboard.py    # Streamlit dashboard
├── 🎲 simulasi.py              # Database simulator  
├── 🎮 run_system.py            # Master controller
├── 🧪 robustness_test.py       # Comprehensive testing
└── 📋 requirements.txt         # Dependencies
```

## 🚀 Cara Menjalankan Sistem

### 1. Persiapan Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Setup MySQL database
mysql -u root -p
CREATE DATABASE sensor_data;
```

### 2. Menjalankan Sistem Lengkap

```bash
# Opsi 1: Jalankan semua komponen
python run_system.py

# Opsi 2: Jalankan komponen individual
python simulasi.py                # Data simulator
python realtime_processor.py     # AI processing
streamlit run realtime_dashboard.py  # Dashboard
```

### 3. Testing Robustness

```bash
python robustness_test.py
```

## 🎛️ Konfigurasi Database

```python
# Database settings di database_manager.py
DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': '',  # Sesuaikan password
    'database': 'sensor_data'
}
```

## 🧠 Model AI Details

### 1. CNN Model
- **Input**: Time series sequence data
- **Architecture**: 
  - Conv1D layers dengan BatchNormalization
  - GlobalAveragePooling untuk dimensionality reduction
  - Dense layers untuk output
- **Use Case**: Pattern recognition dalam sensor data

### 2. LSTM Model  
- **Architecture**:
  - Multi-layer LSTM dengan dropout
  - Bidirectional layers untuk context
  - Dense output dengan multi-task learning
- **Use Case**: Temporal dependencies dan long-term patterns

### 3. Hybrid CNN-LSTM
- **Architecture**:
  - CNN untuk feature extraction
  - LSTM untuk temporal modeling
  - Attention mechanism
  - Multi-task output (regression + classification)
- **Use Case**: Best of both worlds

## 📊 Features Engineering

Sistem menghasilkan **40+ engineered features**:

### Time-based Features
- Jam, hari, bulan patterns
- Lag features (1-5 timesteps)
- Rolling statistics (mean, std, min, max)

### Sensor-specific Features
- pH dan suhu interaction terms
- Rate of change features
- Quality encoding dan statistics

### Advanced Features
- Moving averages (short & long term)
- Seasonal decomposition
- Anomaly detection scores

## 🎯 Real-time Processing Pipeline

1. **Data Ingestion**: MySQL database dengan timestamp
2. **Feature Engineering**: 40+ features dari raw data
3. **Scaling**: MinMaxScaler dan StandardScaler
4. **Sequence Preparation**: Sliding window untuk time series
5. **AI Prediction**: Ensemble dari 3 models
6. **Real-time Display**: Streamlit dashboard update

## 📈 Dashboard Features

### Real-time Charts
- Live sensor readings (pH, temperature)
- Quality predictions
- Model comparison metrics

### Performance Monitoring
- Model accuracy tracking
- Processing latency metrics
- System health indicators

### Interactive Controls
- Time range selection
- Model toggle on/off
- Real-time data refresh

## 🧪 Robustness Testing

### Test Categories

1. **Data Quality Robustness**
   - Missing values handling (5%-20%)
   - Outliers robustness (1%-15%)
   - Data corruption scenarios

2. **Model Performance**
   - Training stability
   - Inference speed testing
   - Cross-validation scores

3. **Noise Robustness**
   - Gaussian noise injection (1%-30%)
   - Signal-to-noise ratio testing
   - Performance degradation analysis

4. **Temporal Robustness**
   - Variable sequence lengths
   - Time lag sensitivity
   - Pattern shift adaptation

### Robustness Grades
- **A+ (0.9+)**: Excellent robustness
- **A (0.8-0.9)**: Very good robustness
- **B+ (0.7-0.8)**: Good robustness  
- **B (0.6-0.7)**: Acceptable robustness
- **C+ (0.5-0.6)**: Fair robustness
- **C (0.4-0.5)**: Poor robustness
- **D (<0.4)**: Very poor robustness

## ⚙️ System Configuration

### Database Health Monitoring
```python
# Automatic health checks every 30 seconds
- Connection status
- Query performance metrics
- Data quality indicators
- Storage utilization
```

### AI Model Configuration
```python
# Model hyperparameters
SEQUENCE_LENGTH = 60        # 60 timesteps input
BATCH_SIZE = 32            # Training batch size
EPOCHS = 50                # Training epochs
LEARNING_RATE = 0.001      # Adam optimizer
DROPOUT_RATE = 0.3         # Regularization
```

### Real-time Processing
```python
# Processing intervals
DATA_FETCH_INTERVAL = 1     # 1 second
PREDICTION_INTERVAL = 5     # 5 seconds  
MODEL_RETRAIN_HOURS = 24    # 24 hours
DASHBOARD_UPDATE = 2        # 2 seconds
```

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check MySQL service
   net start mysql
   
   # Verify credentials in database_manager.py
   ```

2. **Import Errors**
   ```bash
   # Install missing packages
   pip install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size in ai_models.py
   BATCH_SIZE = 16  # Instead of 32
   
   # Limit sequence length
   SEQUENCE_LENGTH = 30  # Instead of 60
   ```

4. **Performance Issues**
   ```python
   # Enable GPU if available
   import tensorflow as tf
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   
   # Optimize database queries
   # Add indexes on timestamp columns
   ```

## 📝 API Endpoints (Future Enhancement)

```python
# Planned REST API endpoints
GET  /api/sensor/latest      # Latest sensor readings
GET  /api/predictions        # Current predictions  
POST /api/predict           # Manual prediction request
GET  /api/models/status     # Model health status
GET  /api/robustness        # Robustness test results
```

## 🎯 Performance Metrics

### Expected Performance
- **Data Processing**: <100ms per batch
- **AI Inference**: <50ms per prediction
- **Dashboard Update**: <2 seconds
- **System Latency**: <500ms end-to-end

### Scalability Targets
- **Data Throughput**: 1000+ records/second
- **Concurrent Users**: 10+ dashboard users
- **Model Retraining**: <30 minutes
- **System Uptime**: >99.5%

## 🔄 Future Enhancements

1. **Advanced AI Models**
   - Transformer architecture
   - Graph Neural Networks
   - Reinforcement Learning

2. **System Improvements**
   - Kubernetes deployment
   - Message queue (Redis/RabbitMQ)
   - Microservices architecture

3. **Monitoring & Alerting**
   - Prometheus metrics
   - Grafana dashboards
   - Slack/email notifications

4. **Data Pipeline**
   - Apache Kafka streaming
   - Apache Spark processing
   - Data lake integration

## 📞 Support & Contact

Untuk pertanyaan atau masalah:
1. Check troubleshooting section
2. Review log files di `logs/` directory
3. Run robustness test untuk diagnosis
4. Check system health di dashboard

---

**Sistem Real-time Sensor AI** - Powered by CNN, LSTM, dan Hybrid Models 🚀
