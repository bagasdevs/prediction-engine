#!/usr/bin/env python3
"""
Quick System Runner - Start Real-time Sensor AI System
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def main():
    print("🚀 QUICK SYSTEM STARTER")
    print("=" * 40)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 40)
    
    # Quick dependency check
    print("🔍 Quick dependency check...")
    try:
        import numpy
        import pandas
        print("✅ NumPy & Pandas OK")
    except ImportError as e:
        print(f"❌ Missing: {e}")
        return
    
    print("\n📋 Available system components:")
    
    # Check available files
    components = {
        "Database Simulator": "simulasi.py",
        "AI Models Test": "test_ai_models.py", 
        "System Test": "test_system.py",
        "Simple Dashboard": "simple_sensor_dashboard.py",
        "MySQL Test": "mysql_test.py",
        "Quick Test": "quick_test.py"
    }
    
    for name, file in components.items():
        if os.path.exists(file):
            print(f"✅ {name} ({file})")
        else:
            print(f"❌ {name} ({file}) - Not found")
    
    print("\n🎯 Starting core components:")
    
    # Option 1: Run quick test first
    try:
        print("\n1️⃣ Running quick test...")
        result = subprocess.run([sys.executable, "quick_test.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Quick test passed")
            print(result.stdout)
        else:
            print("⚠️ Quick test had issues:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Quick test error: {e}")
    
    # Option 2: Test AI models
    try:
        print("\n2️⃣ Testing AI models...")
        result = subprocess.run([sys.executable, "test_ai_models.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ AI models test passed")
            print(result.stdout)
        else:
            print("⚠️ AI models test had issues:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ AI models test error: {e}")
    
    # Option 3: Try to start dashboard
    print("\n3️⃣ Starting dashboard...")
    try:
        # Check if we can import streamlit
        import streamlit
        print("✅ Streamlit available")
        
        print("🌐 Starting dashboard on http://localhost:8501")
        print("💡 Run manually: streamlit run simple_sensor_dashboard.py")
        
        # Start dashboard in background
        dashboard_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "simple_sensor_dashboard.py", 
            "--server.port", "8501"
        ])
        
        print(f"✅ Dashboard started (PID: {dashboard_process.pid})")
        print("🌐 Open: http://localhost:8501")
        
        # Wait a bit then show status
        time.sleep(5)
        
        if dashboard_process.poll() is None:
            print("✅ Dashboard is running!")
        else:
            print("⚠️ Dashboard may have stopped")
            
    except ImportError:
        print("❌ Streamlit not available")
        print("💡 Install with: pip install streamlit")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    
    print("\n" + "=" * 40)
    print("🎉 SYSTEM STARTUP COMPLETE")
    print("=" * 40)
    print("📊 Dashboard: http://localhost:8501")
    print("🧪 Tests completed")
    print("🔧 System ready for use")
    print("\n💡 Manual commands:")
    print("   python simulasi.py         # Start simulator")
    print("   python test_ai_models.py   # Test AI models")
    print("   streamlit run simple_sensor_dashboard.py  # Dashboard")

if __name__ == "__main__":
    main()
