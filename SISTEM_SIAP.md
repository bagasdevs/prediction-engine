# ✅ ML ENGINE SYSTEM - READY TO USE!

## 🎯 **Sistem Saat Ini**

Sistem **Machine Learning Engine** untuk IoT sensor data sudah siap:

```
IoT Devices → MySQL Database → ML Engine → Real-time Dashboard
    ↓             ↓              ↓            ↓
 Auto Feed    sensor_data    AI Models    Visualisasi
```

## 🚀 **Cara Menjalankan**

### 1. **Persiapan Data (Sekali saja)**
```bash
cd data_insert
python setup_initial_data.py
```

### 2. **Jalankan Sistem ML**
```bash
python start_system.py
```

### 3. **Akses Dashboard**
- URL: **http://localhost:8501**
- Auto-refresh setiap 30 detik
- Real-time ML predictions

## 🧠 **Komponen Sistem**

### **ML Engine** (`ml_engine.py`)
- ✅ CNN Model (Convolutional Neural Network)
- ✅ LSTM Model (Long Short-Term Memory)  
- ✅ Hybrid CNN-LSTM Model
- ✅ Real-time predictions setiap 30 detik
- ✅ Automatic retraining setiap jam
- ✅ Background processing

### **ML Dashboard** (`ml_dashboard.py`)
- ✅ Model performance metrics
- ✅ Real-time prediction charts
- ✅ IoT data monitoring
- ✅ Confidence levels visualization

### **Database Integration**
- ✅ MySQL `sensor_data` database
- ✅ Table: `sensor_readings`
- ✅ Auto-populated by IoT devices

## 📊 **Output ML Models**

Sistem menghasilkan 3 prediksi real-time:

1. **CNN Prediction**
   - pH level prediction
   - Temperature prediction  
   - Quality classification
   - Confidence score

2. **LSTM Prediction**
   - Time-series analysis
   - Temporal pattern recognition
   - Sequential data processing

3. **Hybrid Prediction**
   - Combined CNN-LSTM
   - Best of both models
   - Higher accuracy

## 🔧 **Commands**

```bash
# Test sistem
python simple_test.py

# Check komponen
python quick_check.py

# Tambah data manual
cd data_insert && python data_inserter.py

# Check data count
cd data_insert && python data_inserter.py count

# Start sistem
python start_system.py
```

## 💡 **Untuk IoT Devices**

IoT devices harus mengirim data ke MySQL dengan format:

```sql
INSERT INTO sensor_readings (timestamp, ph, suhu, kualitas) 
VALUES (NOW(), 7.2, 25.5, 'baik');
```

## ⚠️ **Troubleshooting**

### Database Error:
```bash
# Start Laragon MySQL service
# Run initial data setup
cd data_insert && python setup_initial_data.py
```

### Dashboard Error:
```bash
# Check if Streamlit installed
pip install streamlit
# Access: http://localhost:8501
```

### ML Engine Error:
```bash
# Check dependencies
pip install -r requirements.txt
```

## 🎉 **SISTEM SIAP!**

- ✅ **No more simulator crashes**
- ✅ **Fokus pada ML output**
- ✅ **Real-time AI predictions**
- ✅ **IoT-ready database**
- ✅ **Automatic model training**

**Happy Machine Learning!** 🤖📊🚀
