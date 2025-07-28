#!/usr/bin/env python3
"""
Complete Real-time Sensor AI System Runner
Menjalankan semua komponen: Simulator, AI Models, dan Dashboard
"""

import subprocess
import time
import threading
import signal
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Import our system components
from ai_system_manager import AISystemManager
from src.database_manager import DatabaseManager

class CompleteSystemRunner:
    """Runner untuk menjalankan semua komponen sistem"""
    
    def __init__(self):
        self.processes = {}
        self.ai_manager = None
        self.is_running = False
        
        # Setup signal handler untuk graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {sig}, shutting down gracefully...")
        self.stop_all_systems()
        sys.exit(0)
    
    def start_simulator(self):
        """Start sensor data simulator"""
        try:
            print("🚀 Starting sensor data simulator...")
            process = subprocess.Popen(
                ["python", "simulasi.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['simulator'] = process
            print(f"✅ Simulator started with PID: {process.pid}")
            return True
        except Exception as e:
            print(f"❌ Error starting simulator: {e}")
            return False
    
    def start_ai_system(self):
        """Start AI system manager"""
        try:
            print("🤖 Starting AI System Manager...")
            self.ai_manager = AISystemManager()
            
            # Load or train models
            try:
                self.ai_manager.ai_models.load_models()
                if not any(self.ai_manager.ai_models.is_trained.values()):
                    print("📚 No trained models found, starting initial training...")
                    success = self.ai_manager.train_initial_models(data_limit=1000)
                    if not success:
                        print("❌ Initial training failed!")
                        return False
                else:
                    print("✅ Pre-trained models loaded successfully")
            except Exception as e:
                print(f"⚠️ Model loading failed: {e}")
                print("📚 Starting fresh training...")
                success = self.ai_manager.train_initial_models(data_limit=1000)
                if not success:
                    print("❌ Training failed!")
                    return False
            
            # Start background tasks
            self.ai_manager.start_background_tasks()
            print("✅ AI System Manager started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting AI system: {e}")
            return False
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        try:
            print("📊 Starting real-time dashboard...")
            process = subprocess.Popen(
                ["streamlit", "run", "simple_sensor_dashboard.py", "--server.port=8501"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['dashboard'] = process
            print(f"✅ Dashboard started with PID: {process.pid}")
            print("🌐 Dashboard available at: http://localhost:8501")
            return True
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            return False
    
    def check_database_connection(self):
        """Check database connection sebelum start"""
        try:
            print("🔍 Checking database connection...")
            db_manager = DatabaseManager()
            if db_manager.connect():
                print("✅ Database connection successful")
                db_manager.disconnect()
                return True
            else:
                print("❌ Database connection failed")
                return False
        except Exception as e:
            print(f"❌ Database check error: {e}")
            return False
    
    def wait_for_data(self, timeout=30):
        """Wait untuk data dari simulator"""
        print(f"⏳ Waiting for sensor data (timeout: {timeout}s)...")
        
        db_manager = DatabaseManager()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if db_manager.connect():
                    query = "SELECT COUNT(*) as count FROM sensor_readings WHERE timestamp > NOW() - INTERVAL 1 MINUTE"
                    result = db_manager.fetch_data(query)
                    
                    if not result.empty and result.iloc[0]['count'] > 0:
                        print(f"✅ Data detected: {result.iloc[0]['count']} recent records")
                        db_manager.disconnect()
                        return True
                    
                    db_manager.disconnect()
            except Exception as e:
                print(f"⚠️ Error checking data: {e}")
            
            time.sleep(2)
        
        print("⚠️ Timeout waiting for data")
        return False
    
    def monitor_system(self):
        """Monitor system health dalam background"""
        while self.is_running:
            try:
                time.sleep(60)  # Check every minute
                
                # Check processes
                dead_processes = []
                for name, process in self.processes.items():
                    if process.poll() is not None:  # Process has terminated
                        dead_processes.append(name)
                
                if dead_processes:
                    print(f"⚠️ Dead processes detected: {dead_processes}")
                
                # Check AI system health
                if self.ai_manager:
                    health = self.ai_manager.get_system_status()
                    if health.get('recent_predictions'):
                        recent = health['recent_predictions']
                        print(f"📈 System alive - Last hour: {recent['count']} predictions")
                
            except Exception as e:
                print(f"❌ Monitor error: {e}")
    
    def run_prediction_demo(self):
        """Demo prediksi real-time"""
        if not self.ai_manager:
            print("❌ AI Manager not initialized")
            return
        
        print("\n🔮 Running prediction demo...")
        
        try:
            predictions = self.ai_manager.predict_realtime_batch(limit=20)
            if predictions:
                print(f"✅ Generated {len(predictions)} predictions")
                
                # Calculate accuracy
                correct_ph = sum(1 for p in predictions if abs(p['ph_predicted'] - p['actual_ph']) < 0.5)
                correct_suhu = sum(1 for p in predictions if abs(p['suhu_predicted'] - p['actual_suhu']) < 2.0)
                correct_kualitas = sum(1 for p in predictions if p['kualitas_predicted'] == p['actual_kualitas'])
                
                print(f"📊 Accuracy:")
                print(f"   pH: {correct_ph}/{len(predictions)} ({correct_ph/len(predictions)*100:.1f}%)")
                print(f"   Suhu: {correct_suhu}/{len(predictions)} ({correct_suhu/len(predictions)*100:.1f}%)")
                print(f"   Kualitas: {correct_kualitas}/{len(predictions)} ({correct_kualitas/len(predictions)*100:.1f}%)")
                
                # Show sample predictions
                print(f"\\n📋 Sample Predictions:")
                for i, pred in enumerate(predictions[:3]):
                    print(f"   Sample {i+1}:")
                    print(f"     pH: {pred['ph_predicted']:.2f} vs {pred['actual_ph']:.2f}")
                    print(f"     Suhu: {pred['suhu_predicted']:.1f} vs {pred['actual_suhu']:.1f}")
                    print(f"     Kualitas: {pred['kualitas_predicted']} vs {pred['actual_kualitas']}")
                    print(f"     Confidence: {pred['confidence']:.3f}")
            else:
                print("⚠️ No predictions generated")
                
        except Exception as e:
            print(f"❌ Error in prediction demo: {e}")
    
    def start_complete_system(self):
        """Start semua komponen sistem"""
        print("🚀 Starting Complete Real-time Sensor AI System")
        print("=" * 60)
        
        # 1. Check database
        if not self.check_database_connection():
            print("❌ Database check failed. Please ensure MySQL is running and database exists.")
            return False
        
        # 2. Start simulator
        if not self.start_simulator():
            print("❌ Failed to start simulator")
            return False
        
        # 3. Wait for initial data
        if not self.wait_for_data(timeout=30):
            print("❌ No data detected from simulator")
            self.stop_all_systems()
            return False
        
        # 4. Start AI system
        if not self.start_ai_system():
            print("❌ Failed to start AI system")
            self.stop_all_systems()
            return False
        
        # 5. Start dashboard
        if not self.start_dashboard():
            print("❌ Failed to start dashboard")
            self.stop_all_systems()
            return False
        
        self.is_running = True
        
        # 6. Start system monitor
        monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        monitor_thread.start()
        
        print("\\n🎉 Complete system started successfully!")
        print("📊 Dashboard: http://localhost:8501")
        print("🤖 AI system monitoring active")
        print("📡 Sensor simulation running")
        print("\\n💡 System Status:")
        
        # Show system status
        if self.ai_manager:
            status = self.ai_manager.get_system_status()
            for key, value in status.items():
                if key not in ['ai_health', 'recent_predictions']:
                    print(f"   {key}: {value}")
        
        return True
    
    def stop_all_systems(self):
        """Stop semua sistem"""
        print("\\n🛑 Stopping all systems...")
        
        self.is_running = False
        
        # Stop AI manager background tasks
        if self.ai_manager:
            self.ai_manager.stop_background_tasks()
            print("✅ AI system stopped")
        
        # Stop all processes
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Still running
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:  # Still running, force kill
                        process.kill()
                    print(f"✅ {name} stopped")
            except Exception as e:
                print(f"⚠️ Error stopping {name}: {e}")
        
        print("✅ All systems stopped")

def main():
    """Main function"""
    print("🌟 Real-time Sensor AI System")
    print("=" * 40)
    print("Components:")
    print("  📡 Sensor Data Simulator")
    print("  🤖 AI Prediction Models (CNN, LSTM, Hybrid)")
    print("  📊 Real-time Dashboard")
    print("  💾 MySQL Database")
    print("=" * 40)
    
    runner = CompleteSystemRunner()
    
    try:
        success = runner.start_complete_system()
        
        if success:
            print("\\n🎯 Running prediction demo...")
            time.sleep(10)  # Wait untuk sistem stabil
            runner.run_prediction_demo()
            
            print("\\n🔄 System is now running continuously...")
            print("💡 Press Ctrl+C to stop the system")
            
            # Keep running until interrupted
            while True:
                time.sleep(30)
                
                # Show periodic status
                if runner.ai_manager:
                    status = runner.ai_manager.get_system_status()
                    if status.get('recent_predictions'):
                        recent = status['recent_predictions']
                        print(f"📈 [{datetime.now().strftime('%H:%M:%S')}] Active - {recent['count']} predictions/hour, confidence: {recent['avg_confidence']:.3f}")
        
        else:
            print("❌ Failed to start complete system")
            return 1
    
    except KeyboardInterrupt:
        print("\\n👋 Shutdown requested by user")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        return 1
    finally:
        runner.stop_all_systems()
        print("🏁 System shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit(main())
