#!/usr/bin/env python3
"""
Standalone Data Inserter untuk Sensor Database
Terpisah dari sistem utama untuk menghindari konflik
"""

import mysql.connector
from mysql.connector import Error
import numpy as np
import random
import time
from datetime import datetime, timedelta
import sys
import os

class SensorDataInserter:
    """Class untuk insert data sensor tanpa konflik dengan sistem utama"""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'sensor_data',
            'port': 3306
        }
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Koneksi ke database"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("✅ Connected to database")
            return True
        except Error as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Tutup koneksi"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("✅ Disconnected from database")
    
    def generate_realistic_data(self):
        """Generate data sensor yang realistis"""
        # pH: 6.0-8.5 (normal range untuk air)
        ph = round(random.uniform(5.5, 9.0), 2)
        
        # Temperature: 18-35°C
        suhu = round(random.uniform(18.0, 35.0), 2)
        
        # Quality berdasarkan pH dan suhu
        if 6.5 <= ph <= 8.0 and 20 <= suhu <= 30:
            kualitas = 'baik'
        elif (6.0 <= ph < 6.5 or 8.0 < ph <= 8.5) and (18 <= suhu < 20 or 30 < suhu <= 32):
            kualitas = 'baik'
        else:
            kualitas = 'buruk'
        
        return ph, suhu, kualitas
    
    def insert_single_record(self):
        """Insert satu record ke database"""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            ph, suhu, kualitas = self.generate_realistic_data()
            
            query = """
            INSERT INTO sensor_readings (ph, suhu, kualitas) 
            VALUES (%s, %s, %s)
            """
            
            self.cursor.execute(query, (ph, suhu, kualitas))
            self.connection.commit()
            
            print(f"✅ Inserted: pH={ph}, Temp={suhu}°C, Quality={kualitas}")
            return True
            
        except Error as e:
            print(f"❌ Insert error: {e}")
            self.connection.rollback()
            return False
    
    def insert_batch_records(self, count=10):
        """Insert multiple records sekaligus"""
        if not self.connection:
            if not self.connect():
                return False
        
        try:
            query = """
            INSERT INTO sensor_readings (ph, suhu, kualitas) 
            VALUES (%s, %s, %s)
            """
            
            data_batch = []
            for _ in range(count):
                ph, suhu, kualitas = self.generate_realistic_data()
                data_batch.append((ph, suhu, kualitas))
            
            self.cursor.executemany(query, data_batch)
            self.connection.commit()
            
            print(f"✅ Inserted {count} records successfully")
            return True
            
        except Error as e:
            print(f"❌ Batch insert error: {e}")
            self.connection.rollback()
            return False
    
    def get_current_count(self):
        """Get jumlah record saat ini"""
        if not self.connection:
            if not self.connect():
                return 0
        
        try:
            self.cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            count = self.cursor.fetchone()[0]
            return count
        except Error as e:
            print(f"❌ Count error: {e}")
            return 0
    
    def continuous_insert(self, interval=5, duration_minutes=10):
        """Insert data secara continuous untuk testing"""
        if not self.connect():
            return
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        inserted_count = 0
        
        print(f"🚀 Starting continuous insert for {duration_minutes} minutes...")
        print(f"📊 Insert interval: {interval} seconds")
        print("-" * 50)
        
        try:
            while time.time() < end_time:
                if self.insert_single_record():
                    inserted_count += 1
                
                time.sleep(interval)
                
                # Show progress every 10 inserts
                if inserted_count % 10 == 0:
                    current_count = self.get_current_count()
                    print(f"📈 Progress: {inserted_count} inserted, Total DB: {current_count}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Insert stopped by user")
        
        finally:
            self.disconnect()
            final_count = self.get_current_count()
            print("-" * 50)
            print(f"🎉 Insert completed!")
            print(f"📊 Records inserted this session: {inserted_count}")
            print(f"📊 Total records in database: {final_count}")

def main():
    """Main function"""
    print("📊 SENSOR DATA INSERTER")
    print("=" * 40)
    print("⚠️  WARNING: Jangan jalankan bersamaan dengan sistem utama!")
    print("💡 Gunakan tool ini untuk populate database secara terpisah")
    print("-" * 40)
    
    inserter = SensorDataInserter()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "single":
            # Insert single record
            if inserter.connect():
                inserter.insert_single_record()
                inserter.disconnect()
        
        elif command == "batch":
            # Insert batch records
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            if inserter.connect():
                inserter.insert_batch_records(count)
                inserter.disconnect()
        
        elif command == "continuous":
            # Continuous insert
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            inserter.continuous_insert(interval, duration)
        
        elif command == "count":
            # Show current count
            if inserter.connect():
                count = inserter.get_current_count()
                print(f"📊 Current records in database: {count}")
                inserter.disconnect()
        
        else:
            print("❌ Unknown command")
            print("💡 Usage:")
            print("   python data_inserter.py single           # Insert 1 record")
            print("   python data_inserter.py batch [count]    # Insert batch records")
            print("   python data_inserter.py continuous [interval] [minutes]  # Continuous insert")
            print("   python data_inserter.py count            # Show current count")
    
    else:
        # Interactive mode
        print("\n🎯 Select operation:")
        print("1. Insert single record")
        print("2. Insert batch records")
        print("3. Continuous insert (for testing)")
        print("4. Show current count")
        print("5. Exit")
        
        while True:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                if inserter.connect():
                    inserter.insert_single_record()
                    inserter.disconnect()
            
            elif choice == "2":
                count = input("Enter number of records (default 20): ").strip()
                count = int(count) if count.isdigit() else 20
                if inserter.connect():
                    inserter.insert_batch_records(count)
                    inserter.disconnect()
            
            elif choice == "3":
                interval = input("Enter interval in seconds (default 5): ").strip()
                interval = int(interval) if interval.isdigit() else 5
                duration = input("Enter duration in minutes (default 10): ").strip()
                duration = int(duration) if duration.isdigit() else 10
                inserter.continuous_insert(interval, duration)
            
            elif choice == "4":
                if inserter.connect():
                    count = inserter.get_current_count()
                    print(f"📊 Current records in database: {count}")
                    inserter.disconnect()
            
            elif choice == "5":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")

if __name__ == "__main__":
    main()
