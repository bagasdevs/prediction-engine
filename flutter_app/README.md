# Flutter Prediction Engine

Aplikasi mobile untuk monitoring sensor real-time dengan prediksi AI.

## 🚀 Quick Start

**Prerequisites:** Flutter SDK, Python 3.7+, MySQL

**Setup:**
```cmd
# 1. Setup database
scripts\setup_database.bat

# 2. Start backend
scripts\start_api.bat

# 3. Start simulator
scripts\start_simulator.bat

# 4. Run Flutter app
scripts\start_flutter.bat
```

## 📱 Features

- Real-time monitoring (auto-refresh 5s)
- Manual prediction input
- Live data indicators
- Quality recommendations

## 🔧 Troubleshooting

- **Flutter not found:** Tambahkan Flutter ke PATH
- **API error:** Pastikan Python API running di port 5000
- **Database error:** Pastikan MySQL running (Laragon/XAMPP)
