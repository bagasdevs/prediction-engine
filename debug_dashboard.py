#!/usr/bin/env python3
"""
Debug Dashboard Database Connection Issue
"""

import sys
import os

# Add src to path (same as dashboard)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 DEBUGGING DASHBOARD DATABASE CONNECTION")
print("=" * 50)

# Test 1: Import DatabaseManager
print("\n1️⃣ Testing DatabaseManager import...")
try:
    from src.database_manager import DatabaseManager
    print("✅ DatabaseManager imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create DatabaseManager instance
print("\n2️⃣ Creating DatabaseManager instance...")
try:
    db = DatabaseManager()
    print("✅ DatabaseManager instance created")
    print(f"📋 Config: {db.config}")
except Exception as e:
    print(f"❌ Instance creation error: {e}")
    sys.exit(1)

# Test 3: Test connection
print("\n3️⃣ Testing database connection...")
try:
    connection_result = db.connect()
    print(f"🔗 Connection result: {connection_result}")
    
    if connection_result:
        print("✅ Connection successful!")
        
        # Test cursor
        if hasattr(db, 'cursor') and db.cursor:
            print("✅ Cursor available")
            
            # Test query
            try:
                db.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
                count = db.cursor.fetchone()[0]
                print(f"📊 Records found: {count}")
                
                db.disconnect()
                print("✅ Disconnection successful")
                
            except Exception as e:
                print(f"❌ Query error: {e}")
        else:
            print("❌ Cursor not available")
    else:
        print("❌ Connection failed")
        
except Exception as e:
    print(f"❌ Connection test error: {e}")

# Test 4: Test the exact same function as dashboard
print("\n4️⃣ Testing dashboard connection function...")
def test_database_connection():
    """Same function as in dashboard"""
    try:
        db = DatabaseManager() 
        if db.connect():
            db.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            count = db.cursor.fetchone()[0]
            db.disconnect()
            return True, count
        else:
            return False, "Connection failed"
    except Exception as e:
        return False, str(e)

try:
    conn_ok, result = test_database_connection()
    print(f"🧪 Dashboard test result: {conn_ok}, {result}")
    
    if conn_ok:
        print("✅ Dashboard function works - problem elsewhere!")
    else:
        print(f"❌ Dashboard function failed: {result}")
        
except Exception as e:
    print(f"❌ Dashboard function error: {e}")

print("\n" + "=" * 50)
if conn_ok:
    print("🎉 DATABASE CONNECTION WORKING!")
    print("💡 Problem might be in Streamlit caching or session state")
    print("🔄 Try clearing Streamlit cache or restart dashboard")
else:
    print("❌ DATABASE CONNECTION STILL FAILING")
    print("💡 Need to fix database configuration")
