#!/usr/bin/env python3
"""
Realtime Data Simulator - Simple Version
Generate data sensor otomatis untuk Flutter app
"""

import sys
import os
import subprocess
import time

# Add project to path
sys.path.append(os.path.dirname(__file__))

def run_simulasi1():
    """Run simulasi1.py dengan interval 5 detik"""
    print("🚀 Starting Real-time Sensor Data Generation")
    print("=" * 50)
    print("📊 Data akan di-generate setiap 5 detik")
    print("📱 Flutter app akan menampilkan data real-time")
    print("🔄 Tekan Ctrl+C untuk stop")
    print("=" * 50)
    
    try:
        # Run simulasi1.py dengan interval 5 detik
        result = subprocess.run([
            sys.executable, 
            os.path.join('simulasi_data', 'simulasi1.py'), 
            '5'  # 5 second interval
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Data generation stopped")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_fast_mode():
    """Run dengan interval 1 detik untuk testing"""
    print("🚀 Fast Mode: 1 second interval")
    
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join('simulasi_data', 'simulasi1.py'), 
            '1'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Fast mode stopped")

def generate_sample_data():
    """Generate sample data batch"""
    print("📦 Generating sample data...")
    
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join('simulasi_data', 'simulasi1.py'), 
            '--batch', '50'
        ])
        
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'fast':
            run_fast_mode()
        elif command == 'sample':
            generate_sample_data()
        else:
            print("❌ Unknown command")
            print("💡 Available commands:")
            print("   python realtime_simulator.py fast")
            print("   python realtime_simulator.py sample")
    else:
        # Default: 5 second interval
        run_simulasi1()

if __name__ == '__main__':
    main()
