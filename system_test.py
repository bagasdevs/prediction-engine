#!/usr/bin/env python3
"""
Comprehensive System Test - Test semua komponen sistem
"""

import sys
import os
from datetime import datetime

def test_imports():
    """Test semua imports"""
    print("🔍 Testing imports...")
    
    try:
        # Database manager
        from src.database_manager import DatabaseManager
        print("✅ DatabaseManager imported")
        
        # Preprocessor
        from src.realtime_preprocessor import SensorDataPreprocessor
        print("✅ SensorDataPreprocessor imported")
        
        # AI Models
        from src.ai_models import SensorAIModels
        print("✅ SensorAIModels imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database():
    """Test database connection"""
    print("\n💾 Testing database...")
    
    try:
        from src.database_manager import DatabaseManager
        
        db = DatabaseManager()
        if db.connect():
            print("✅ Database connection successful")
            
            # Test basic query
            result = db.fetch_data("SELECT COUNT(*) as count FROM sensor_readings")
            if not result.empty:
                count = result.iloc[0]['count']
                print(f"✅ Found {count} records in database")
                db.disconnect()
                return True
            else:
                print("❌ Query failed")
                return False
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test error: {e}")
        return False

def test_ai_models():
    """Test AI models"""
    print("\n🤖 Testing AI models...")
    
    try:
        from src.ai_models import SensorAIModels
        import numpy as np
        
        # Create small test case
        timesteps, features = 20, 10
        ai_models = SensorAIModels(input_shape=(timesteps, features))
        
        # Build one model
        cnn_model = ai_models.build_cnn_model()
        print(f"✅ CNN model built successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ AI models test error: {e}")
        return False

def run_full_test():
    """Run all tests"""
    print("🚀 Comprehensive System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Database", test_database),
        ("AI Models", test_ai_models)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:15}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ System is ready to run")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Please fix issues before running the complete system")
    
    return all_passed

if __name__ == "__main__":
    run_full_test()
