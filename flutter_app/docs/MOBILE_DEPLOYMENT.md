# 📱 Mobile Device Deployment Guide

## 🚀 Quick Setup

**1. Setup Network & Firewall**
```cmd
scripts\setup_network.bat
```

**2. Deploy to Device**
```cmd
scripts\deploy_to_device.bat
```

## 📋 Step-by-Step Guide

### **A. Android Device Setup**

1. **Enable Developer Options:**
   - Settings → About Phone
   - Tap "Build Number" 7 times
   - Go back → Developer Options

2. **Enable USB Debugging:**
   - ✅ USB Debugging
   - ✅ Install via USB
   - ✅ Stay awake

3. **Connect Device:**
   - USB cable to computer
   - Select "File Transfer" mode
   - Allow USB Debugging (popup)

### **B. Computer Setup**

1. **Check Device Connection:**
   ```cmd
   flutter devices
   ```

2. **Setup Network Access:**
   ```cmd
   # Your computer IP: 192.168.10.246
   # API URL: http://192.168.10.246:5000
   scripts\setup_network.bat
   ```

3. **Start Backend Services:**
   ```cmd
   # Terminal 1: API Server
   scripts\start_api.bat
   
   # Terminal 2: Data Simulator
   scripts\start_simulator.bat
   ```

### **C. Deploy & Test**

1. **Deploy to Device:**
   ```cmd
   scripts\deploy_to_device.bat
   ```

2. **Test API from Device:**
   - Open browser di HP
   - Go to: `http://192.168.10.246:5000/health`
   - Should see: `{"status":"healthy"}`

3. **Test Flutter App:**
   - App should auto-refresh data
   - Try manual prediction
   - Check live data indicators

## 🔧 Troubleshooting

### **Device Not Found:**
- Check USB cable
- Re-enable USB Debugging
- Try different USB port
- Install Android drivers

### **Network Connection Failed:**
- Run `scripts\setup_network.bat`
- Check Windows Firewall
- Ensure both devices on same WiFi
- Test API URL in browser first

### **App Crashes:**
- Check `flutter doctor`
- Clear app data on device
- Rebuild: `flutter clean && flutter build apk`

### **API Not Accessible:**
```cmd
# Check if API running
curl http://192.168.10.246:5000/health

# Add firewall rule manually
netsh advfirewall firewall add rule name="Flutter API" dir=in action=allow protocol=TCP localport=5000
```

## 📱 Alternative Testing

### **Build APK (No USB):**
```cmd
flutter build apk --release
# APK location: build\app\outputs\flutter-apk\app-release.apk
```

### **Wireless Debugging (Android 11+):**
```cmd
# Enable Wireless Debugging in Developer Options
adb pair [IP:PORT]
adb connect [IP:PORT]
flutter run
```

## 🎯 Production Deployment

### **Deploy API to Cloud:**
- Heroku, Railway, atau Google Cloud
- Update API URL di `main.dart`
- Build production APK

### **Local Network Only:**
- Keep current setup (192.168.10.246:5000)
- Share APK to other devices on same network
- Ensure firewall allows connections
