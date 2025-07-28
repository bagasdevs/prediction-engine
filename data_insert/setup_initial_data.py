#!/usr/bin/env python3
"""
Setup Initial Data untuk Database
Populate database dengan data awal yang cukup untuk AI training
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from data_inserter import SensorDataInserter
import time

def populate_initial_data():
    """Populate database dengan data awal"""
    print("🚀 INITIAL DATA SETUP")
    print("=" * 50)
    
    inserter = SensorDataInserter()
    
    if not inserter.connect():
        print("❌ Cannot connect to database")
        return False
    
    # Check current count
    current_count = inserter.get_current_count()
    print(f"📊 Current records in database: {current_count}")
    
    # Define target counts
    minimum_for_ai = 500  # Minimum untuk AI training
    target_count = 1000   # Target optimal
    
    if current_count >= target_count:
        print(f"✅ Database sudah memiliki cukup data ({current_count} records)")
        print("💡 Tidak perlu menambah data")
        inserter.disconnect()
        return True
    
    needed_count = target_count - current_count
    print(f"📈 Need to add {needed_count} records to reach target")
    
    # Insert in batches
    batch_size = 50
    batches = (needed_count + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"📦 Will insert in {batches} batches of {batch_size} records each")
    print("-" * 50)
    
    try:
        for i in range(batches):
            current_batch_size = min(batch_size, needed_count - (i * batch_size))
            
            print(f"📦 Batch {i+1}/{batches}: Inserting {current_batch_size} records...")
            
            if inserter.insert_batch_records(current_batch_size):
                print(f"✅ Batch {i+1} completed")
            else:
                print(f"❌ Batch {i+1} failed")
                return False
            
            # Small delay between batches
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n⏹️ Setup interrupted by user")
        return False
    
    finally:
        inserter.disconnect()
    
    # Final verification
    inserter.connect()
    final_count = inserter.get_current_count()
    inserter.disconnect()
    
    print("-" * 50)
    print("🎉 INITIAL DATA SETUP COMPLETE!")
    print(f"📊 Final count: {final_count} records")
    
    if final_count >= minimum_for_ai:
        print("✅ Database ready for AI training")
    else:
        print("⚠️ May need more data for optimal AI performance")
    
    return True

def create_sample_historical_data():
    """Create sample data dengan timestamp yang bervariasi"""
    print("\n📅 Creating historical data...")
    
    inserter = SensorDataInserter()
    if not inserter.connect():
        return False
    
    try:
        # Create data dengan timestamp yang spread over time
        historical_query = """
        INSERT INTO sensor_readings (ph, suhu, kualitas, timestamp) 
        VALUES (%s, %s, %s, %s)
        """
        
        from datetime import datetime, timedelta
        import random
        
        historical_data = []
        base_time = datetime.now() - timedelta(days=7)  # 7 days ago
        
        # Generate 100 historical records
        for i in range(100):
            # Random time within last 7 days
            random_minutes = random.randint(0, 7 * 24 * 60)  # 7 days in minutes
            timestamp = base_time + timedelta(minutes=random_minutes)
            
            # Generate sensor data
            ph, suhu, kualitas = inserter.generate_realistic_data()
            
            historical_data.append((ph, suhu, kualitas, timestamp))
        
        inserter.cursor.executemany(historical_query, historical_data)
        inserter.connection.commit()
        
        print("✅ Added 100 historical records")
        return True
        
    except Exception as e:
        print(f"❌ Historical data error: {e}")
        return False
    
    finally:
        inserter.disconnect()

def main():
    """Main setup function"""
    print("🏗️ DATABASE INITIAL SETUP")
    print("=" * 50)
    print("This will populate the database with initial data")
    print("for AI training and testing purposes.")
    print("-" * 50)
    
    # Step 1: Populate initial data
    if not populate_initial_data():
        print("\n❌ Initial data setup failed")
        return
    
    # Step 2: Create historical data
    if not create_sample_historical_data():
        print("\n❌ Historical data setup failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 DATABASE SETUP COMPLETE!")
    print("=" * 50)
    print("✅ Database populated with training data")
    print("✅ Historical data created")
    print("✅ Ready for AI system")
    
    print("\n🚀 Next steps:")
    print("1. Run AI system: python start_system.py")
    print("2. Add more data: cd data_insert && python data_inserter.py")
    print("3. Monitor: streamlit run fixed_dashboard.py")

if __name__ == "__main__":
    main()
