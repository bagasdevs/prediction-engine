#!/usr/bin/env python3
"""
Test Dashboard Connection
"""

import sys
import os
sys.path.append('src')

def test_database_connection():
    """Test database connection dari dashboard perspective"""
    try:
        from src.database_manager import DatabaseManager
        
        print("🔍 Testing database connection...")
        db = DatabaseManager()
        
        if db.connect():
            print("✅ Database connection successful!")
            
            # Get recent data
            recent_data = db.get_latest_data(10)
            print(f"📊 Found {len(recent_data)} recent records")
            
            if len(recent_data) > 0:
                print("📋 Sample data:")
                print(recent_data.head())
            
            db.disconnect()
            return True
        else:
            print("❌ Database connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🧪 DASHBOARD CONNECTION TEST")
    print("=" * 40)
    
    success = test_database_connection()
    
    if success:
        print("\n✅ Database connection working!")
        print("🎉 Dashboard should now work properly")
        print("\n🚀 To start dashboard:")
        print("   streamlit run simple_sensor_dashboard.py")
    else:
        print("\n❌ Database connection still has issues")
        print("💡 Try running: python fix_database.py")

if __name__ == "__main__":
    main()
