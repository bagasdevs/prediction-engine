#!/usr/bin/env python3
"""
ML Engine System Starter
Real-time Sensor AI System fokus pada Machine Learning Output
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def display_system_banner():
    """Display system startup banner"""
    print("🚀 REAL-TIME SENSOR AI SYSTEM")
    print("=" * 50)
    print("🤖 CNN, LSTM, Hybrid AI Models")
    print("📊 Real-time Dashboard")  
    print("🗄️ MySQL Database (IoT Data Source)")
    print("🧠 Machine Learning Prediction Engine")
    print("📈 Real-time AI Output Generation")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Data fed by IoT devices automatically")
    print("-" * 50)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking system dependencies...")
    
    # Check Python modules
    required_modules = [
        'numpy', 'pandas', 'mysql.connector', 
        'tensorflow', 'sklearn', 'streamlit'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}")
    
    if missing_modules:
        print(f"\n⚠️ Missing modules: {', '.join(missing_modules)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def start_dashboard():
    """Start ML Output Dashboard"""
    print("\n📊 Starting ML Output Dashboard...")
    try:
        # Priority order for dashboards
        dashboard_files = ["ml_dashboard.py", "fixed_dashboard.py", "simple_sensor_dashboard.py"]
        dashboard_file = None
        
        for file in dashboard_files:
            if os.path.exists(file):
                dashboard_file = file
                break
        
        if not dashboard_file:
            print("❌ No dashboard file found")
            return None
        
        print(f"📊 Using dashboard: {dashboard_file}")
        
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", dashboard_file, "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✅ ML Dashboard started (PID: {process.pid})")
        print("🌐 Access: http://localhost:8501")
        print("📊 Real-time ML output visualization ready")
        return process
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return None

def check_database_data():
    """Check if database has enough data for AI system"""
    print("\n📊 Checking database data...")
    try:
        import mysql.connector
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        print(f"📊 Found {count} records in database")
        
        if count < 100:
            print("⚠️ Database has insufficient data for AI training")
            print("💡 Run: cd data_insert && python setup_initial_data.py")
            return False
        else:
            print("✅ Database has sufficient data")
            return True
            
    except Exception as e:
        print(f"❌ Database check error: {e}")
        print("💡 Make sure Laragon MySQL is running")
        print("💡 Run: cd data_insert && python setup_initial_data.py")
        return False

def start_ml_engine():
    """Start Machine Learning Engine"""
    print("\n🧠 Starting Machine Learning Engine...")
    try:
        if not os.path.exists("ml_engine.py"):
            print("❌ ml_engine.py not found")
            return None
            
        process = subprocess.Popen(
            [sys.executable, "ml_engine.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✅ ML Engine started (PID: {process.pid})")
        print("📊 Real-time AI processing active")
        print("🔄 Continuous model training scheduled")
        return process
    except Exception as e:
        print(f"❌ ML Engine error: {e}")
        return None

def main():
    """Main system starter"""
    display_system_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ System cannot start due to missing dependencies")
        print("💡 Run: pip install -r requirements.txt")
        return
    
    # Check database data
    if not check_database_data():
        print("\n❌ System cannot start due to insufficient database data")
        print("💡 Please run: cd data_insert && python setup_initial_data.py")
        return
    
    processes = []
    
    try:
        # Start dashboard first
        print("\n🚀 Starting system components...")
        dashboard = start_dashboard()
        if dashboard:
            processes.append(("ML Dashboard", dashboard))
            time.sleep(3)
        else:
            print("⚠️ Dashboard failed to start, continuing with ML Engine only")
        
        # Start ML Engine
        ml_engine = start_ml_engine()
        if ml_engine:
            processes.append(("ML Engine", ml_engine))
        else:
            print("⚠️ ML Engine failed to start")
        
        if processes:
            print("\n" + "=" * 50)
            print("🎉 SYSTEM STARTED SUCCESSFULLY!")
            print("=" * 50)
            
            for name, proc in processes:
                print(f"✅ {name} running (PID: {proc.pid})")
            
            print("\n🌐 Dashboard: http://localhost:8501")
            print("📊 Real-time data monitoring from IoT devices")
            print("🧠 AI models generating predictions continuously")
            print("📈 Machine Learning output updated every 30 seconds")
            print("🔄 Model retraining every hour with new IoT data")
            print("\n💡 IoT devices should feed data automatically to MySQL")
            print("📊 To check data: cd data_insert && python data_inserter.py count")
            print("\n⏹️ Press Ctrl+C to stop all services")
            print("=" * 50)
            
            # Keep running and monitor processes
            try:
                while True:
                    time.sleep(10)
                    # Check if processes are still alive
                    for name, proc in processes[:]:  # Copy list to avoid modification during iteration
                        if proc.poll() is not None:
                            print(f"⚠️ {name} stopped unexpectedly (exit code: {proc.poll()})")
                            processes.remove((name, proc))
                    
                    # If all processes stopped, exit
                    if not processes:
                        print("❌ All processes stopped, exiting...")
                        break
            
            except KeyboardInterrupt:
                print("\n\n🛑 Shutdown requested by user...")
        
        else:
            print("\n❌ No components started successfully")
            print("💡 Check the error messages above for troubleshooting")
            
    except Exception as e:
        print(f"\n❌ System error: {e}")
    
    finally:
        # Cleanup
        if processes:
            print("\n🧹 Stopping all services...")
            for name, proc in processes:
                try:
                    proc.terminate()
                    # Wait for process to terminate gracefully
                    try:
                        proc.wait(timeout=5)
                        print(f"✅ {name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        print(f"✅ {name} force stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping {name}: {e}")
        
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main()
