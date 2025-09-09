# Project Structure

## 📁 Organized Flutter App Structure

```
flutter_app/
├── lib/
│   └── main.dart                    # Flutter mobile app
├── scripts/
│   ├── start_api.bat               # Start Python backend
│   ├── start_flutter.bat           # Start Flutter app
│   ├── start_simulator.bat         # Start data generator
│   └── setup_database.bat          # Database setup
├── backend/
│   ├── flutter_api.py              # API server
│   ├── setup_database.py           # Database setup script
│   └── simulasi1.py                # Data simulator
├── docs/
│   └── (documentation)
├── quick_start.bat                  # One-click launcher
├── pubspec.yaml                     # Flutter dependencies
└── README.md                       # Quick guide
```

## 🚀 Usage

**Quick Start:**
```cmd
cd flutter_app
quick_start.bat
```

**Manual:**
```cmd
# 1. Setup database
scripts\setup_database.bat

# 2. Start backend
scripts\start_api.bat

# 3. Start simulator  
scripts\start_simulator.bat

# 4. Run Flutter
scripts\start_flutter.bat
```

## 🧹 Cleaned Up

**Moved to flutter_app:**
- ✅ Flutter API backend
- ✅ Database setup scripts
- ✅ Data simulator
- ✅ Launch scripts
- ✅ Documentation

**Removed:**
- ❌ Duplicate files
- ❌ Old batch scripts
- ❌ Unused setup files

**Result:** Clean, organized, self-contained Flutter project!
