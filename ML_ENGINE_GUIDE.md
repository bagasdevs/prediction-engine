# ML Engine System - Quick Guide

## 🎯 Sistem Saat Ini

Sistem sekarang **FOKUS PADA MACHINE LEARNING ENGINE** tanpa simulator internal:

```
IoT Devices → MySQL Database → ML Engine → Dashboard
     ↓              ↓              ↓          ↓
   Auto Feed    Sensor Data    AI Models   Visualisasi
```

## 🚀 Cara Menjalankan

### 1. Persiapan Data (Sekali saja)
```bash
cd data_insert
python setup_initial_data.py
```

### 2. Jalankan Sistem ML
```bash
python start_system.py
```

## 📊 Komponen Sistem

1. **ML Engine** (`ml_engine.py`)
   - CNN, LSTM, Hybrid Models
   - Real-time predictions setiap 30 detik
   - Model retraining setiap jam
   - Background processing

2. **ML Dashboard** (`ml_dashboard.py`)
   - Visualisasi AI predictions
   - IoT data monitoring
   - Model performance metrics
   - Real-time updates

3. **Database Manager** (`src/database_manager.py`)
   - MySQL connection handling
   - Data retrieval for ML training
   - Prediction storage

## 🤖 AI Models Output

Sistem menghasilkan 3 jenis prediksi:
- **CNN Prediction**: Convolutional Neural Network
- **LSTM Prediction**: Long Short-Term Memory
- **Hybrid Prediction**: CNN-LSTM combined

## 📈 Dashboard Features

- **Real-time Metrics**: Model confidence, prediction count
- **Prediction Charts**: pH, Temperature, Quality comparisons
- **IoT Monitoring**: Data flow rate, quality distribution
- **Model Performance**: Confidence levels per model

## 💡 Untuk IoT Integration

Database akan diisi otomatis oleh IoT devices ke tabel `sensor_readings`:
```sql
CREATE TABLE sensor_readings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    ph FLOAT,
    suhu FLOAT,
    kualitas VARCHAR(20)
);
```

## 🔧 Troubleshooting

- **Check System**: `python quick_check.py`
- **Test Components**: `python test_system.py`
- **Add Data**: `cd data_insert && python data_inserter.py`
- **Dashboard**: http://localhost:8501

## ⚠️ PENTING

- **JANGAN jalankan simulator bersamaan dengan ML Engine**
- IoT devices harus feed data otomatis ke MySQL
- Database minimal 100 records untuk AI training
- Dashboard auto-refresh setiap 30 detik
