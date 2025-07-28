#!/usr/bin/env python3
"""
Master Controller untuk Real-time Sensor AI System
Mengendalikan semua komponen sistem
"""

import asyncio
import subprocess
import sys
import os
import time
import threading
from pathlib import Path

class SystemController:
    """Controller untuk mengelola semua komponen sistem"""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def start_database_simulator(self):
        """Start database simulator"""
        print("🗄️ Starting database simulator...")
        try:
            process = subprocess.Popen(
                ["C:/Users/ThinkPad T495/AppData/Local/Programs/Python/Python312/python.exe", "simulasi.py"],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['simulator'] = process
            print("✅ Database simulator started")
            return True
        except Exception as e:
            print(f"❌ Error starting simulator: {e}")
            return False
    
    def start_ai_processor(self):
        """Start AI processor"""
        print("🤖 Starting AI processor...")
        try:
            # Import here to avoid circular imports
            sys.path.append('src')
            from src.realtime_processor import RealTimeProcessor
            
            # Create and run processor in separate thread
            def run_processor():
                async def processor_main():
                    processor = RealTimeProcessor(
                        retrain_interval_minutes=30,
                        prediction_interval_seconds=5
                    )
                    
                    try:
                        await processor.initialize()
                        await processor.run_realtime_loop()
                    except Exception as e:
                        print(f"❌ AI Processor error: {e}")
                        
                asyncio.run(processor_main())
            
            thread = threading.Thread(target=run_processor, daemon=True)
            thread.start()
            self.processes['ai_processor'] = thread
            print("✅ AI processor started")
            return True
        except Exception as e:
            print(f"❌ Error starting AI processor: {e}")
            return False
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        print("📊 Starting dashboard...")
        try:
            process = subprocess.Popen(
                ["C:/Users/ThinkPad T495/AppData/Local/Programs/Python/Python312/python.exe", "-m", "streamlit", "run", "realtime_dashboard.py", "--server.port", "8501"],
                cwd=os.getcwd(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes['dashboard'] = process
            print("✅ Dashboard started at http://localhost:8501")
            return True
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            return False
    
    def start_all(self):
        """Start all system components"""
        print("🚀 Starting Real-time Sensor AI System...")
        print("=" * 50)
        
        # Wait a bit between starts
        if self.start_database_simulator():
            time.sleep(3)
            
        if self.start_ai_processor():
            time.sleep(5)
            
        if self.start_dashboard():
            time.sleep(2)
        
        self.running = True
        print("\n" + "=" * 50)
        print("🎉 System started successfully!")
        print("📊 Dashboard: http://localhost:8501")
        print("🗄️ Database simulator running")
        print("🤖 AI processor running")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)
    
    def stop_all(self):
        """Stop all system components"""
        print("\n🛑 Stopping all services...")
        
        for name, process in self.processes.items():
            try:
                if hasattr(process, 'terminate'):
                    process.terminate()
                    print(f"✅ {name} stopped")
                elif hasattr(process, 'join'):
                    print(f"✅ {name} thread will stop")
            except Exception as e:
                print(f"⚠️ Error stopping {name}: {e}")
        
        self.running = False
        print("✅ All services stopped")
    
    def monitor_system(self):
        """Monitor system health"""
        try:
            while self.running:
                time.sleep(30)  # Check every 30 seconds
                
                # Check if processes are still running
                for name, process in list(self.processes.items()):
                    if hasattr(process, 'poll'):
                        if process.poll() is not None:
                            print(f"⚠️ {name} process stopped unexpectedly")
                            self.processes.pop(name)
                
        except KeyboardInterrupt:
            pass

def main():
    """Main function"""
    controller = SystemController()
    
    try:
        # Start all components
        controller.start_all()
        
        # Monitor system
        controller.monitor_system()
        
    except KeyboardInterrupt:
        print("\n⏹️ Shutdown requested by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
    finally:
        controller.stop_all()

if __name__ == "__main__":
    main()
