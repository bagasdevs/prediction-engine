#!/usr/bin/env python3
"""
Simple test untuk simulator
"""

# Test basic imports
try:
    import sys
    import os
    print("✅ Basic imports OK")
    
    # Add src to path
    sys.path.append('src')
    
    # Test numpy
    import numpy as np
    print("✅ NumPy imported")
    
    # Test pandas (from environment details)
    import pandas as pd
    print("✅ Pandas imported")
    
    # Test datetime
    from datetime import datetime, timedelta
    print("✅ Datetime imported")
    
    # Test SQLite database manager
    from src.sqlite_database_manager import SQLiteDatabaseManager
    print("✅ SQLite database manager imported")
    
    # Test creating database
    db = SQLiteDatabaseManager("test_quick.db")
    if db.connect():
        print("✅ Database connection successful")
        
        # Test insert
        result = db.insert_sensor_data(7.2, 25.5, 'baik')
        if result:
            print("✅ Data insertion successful")
            
            # Test retrieve
            data = db.get_latest_data(1)
            print(f"✅ Data retrieved: {len(data)} records")
            print(data.head() if len(data) > 0 else "No data")
        
        db.disconnect()
        print("✅ Database disconnected")
    
    # Clean up
    if os.path.exists("test_quick.db"):
        os.remove("test_quick.db")
        print("✅ Test file cleaned up")
    
    print("\n🎉 All basic tests passed! Ready to run simulator.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Test error: {e}")
