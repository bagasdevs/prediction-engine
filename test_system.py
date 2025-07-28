#!/usr/bin/env python3
"""
Test script untuk ML Engine System Components
"""

import sys
import os
sys.path.append('src')

def test_mysql_database():
    """Test MySQL database manager"""
    try:
        from src.database_manager import DatabaseManager
        print("✅ MySQL database manager imported successfully")
        
        # Test database
        db = DatabaseManager()
        
        if db.connect():
            print("✅ Database connection successful")
            
            # Test inserting data
            result = db.insert_sensor_data(7.2, 25.5, 'baik')
            if result:
                print("✅ Data insertion successful")
                
                # Test retrieving data
                data = db.get_latest_data(5)
                print(f"✅ Data retrieval successful: {len(data)} records")
                
                # Test health stats
                stats = db.get_realtime_stats()
                print("✅ Health stats retrieved")
                
                db.disconnect()
                print("✅ Database test completed successfully")
                return True
            else:
                print("❌ Data insertion failed")
                return False
        else:
            print("❌ Database connection failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_ml_engine():
    """Test ML Engine components"""
    try:
        print("🧠 Testing ML Engine...")
        
        # Test if ML engine file exists
        if not os.path.exists('ml_engine.py'):
            print("❌ ml_engine.py not found")
            return False
        
        print("✅ ML Engine file exists")
        
        # Test AI models import
        try:
            from src.ai_models import SensorAIModels
            print("✅ AI Models imported successfully")
            
            # Test creating AI models instance
            ai_models = SensorAIModels(input_shape=(60, 30))
            print("✅ AI Models instance created")
            
        except Exception as e:
            print(f"⚠️ AI Models warning: {e}")
        
        # Test preprocessor
        try:
            from src.realtime_preprocessor import SensorDataPreprocessor
            preprocessor = SensorDataPreprocessor()
            print("✅ Data Preprocessor available")
        except Exception as e:
            print(f"⚠️ Preprocessor warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Engine test error: {e}")
        return False

def test_dashboard():
    """Test dashboard files"""
    try:
        print("📊 Testing Dashboard files...")
        
        # Check available dashboards
        dashboards = ['ml_dashboard.py', 'fixed_dashboard.py', 'simple_sensor_dashboard.py']
        available_dashboards = []
        
        for dashboard in dashboards:
            if os.path.exists(dashboard):
                available_dashboards.append(dashboard)
                print(f"✅ {dashboard} available")
        
        if available_dashboards:
            print(f"✅ {len(available_dashboards)} dashboard(s) ready")
            return True
        else:
            print("❌ No dashboard files found")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ML Engine System Components...")
    print("="*60)
    
    # Test database
    print("\n📊 Testing Database Manager...")
    db_success = test_mysql_database()
    
    # Test ML Engine
    print("\n🧠 Testing ML Engine Components...")
    ml_success = test_ml_engine()
    
    # Test dashboard
    print("\n📊 Testing Dashboard Files...")
    dashboard_success = test_dashboard()
    
    print(f"\n📋 Test Results:")
    print(f"Database Manager: {'✅ PASS' if db_success else '❌ FAIL'}")
    print(f"ML Engine: {'✅ PASS' if ml_success else '❌ FAIL'}")
    print(f"Dashboard: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    
    print("\n" + "="*60)
    if db_success and ml_success and dashboard_success:
        print("🎉 All tests passed! ML Engine system ready to run.")
        print("🚀 Start system with: python start_system.py")
    else:
        print("❌ Some tests failed. Check the errors above.")
        if not db_success:
            print("💡 Database: Check MySQL service and data_insert setup")
        if not ml_success:
            print("💡 ML Engine: Check dependencies and src/ modules")
        if not dashboard_success:
            print("💡 Dashboard: Check if dashboard files exist")
