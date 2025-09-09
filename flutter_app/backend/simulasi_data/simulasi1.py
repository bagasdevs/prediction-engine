#!/usr/bin/env python3
"""
Enhanced Database Simulator untuk Real-time Sensor AI System
Menghasilkan data sensor yang realistis dengan temporal patterns
Menggunakan MySQL database
"""

import mysql.connector
import random
import time
import numpy as np
from datetime import datetime, timedelta
import signal
import threading
import json
import sys

class SensorDataSimulator:
    def __init__(self):
        # MySQL database configuration
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',  # Sesuaikan dengan password MySQL Anda
            'database': 'sensor_data'
        }
        
        self.connection = None
        self.cursor = None
        
        self.running = True
        self.stats = {
            'total_generated': 0,
            'start_time': None,
            'current_session': 0,
            'errors': 0
        }
        
        # Setup signal handler untuk graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n⏹️ Received signal {signum}. Shutting down gracefully...")
        self.running = False
    
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            
            # Create table if not exists
            self.create_table()
            
            print("✅ Connected to MySQL database")
            return True
                
        except mysql.connector.Error as err:
            print(f"❌ Database connection error: {err}")
            return False
    
    def create_table(self):
        """Create sensor data table if not exists"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                no INT AUTO_INCREMENT PRIMARY KEY,
                ph DECIMAL(5,2) NOT NULL,
                suhu DECIMAL(5,2) NOT NULL,
                kualitas ENUM('baik', 'sedang', 'buruk') NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp)
            )
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("✅ Database table verified/created")
            
        except mysql.connector.Error as err:
            print(f"❌ Error creating table: {err}")
            raise
    
    def generate_sensor_data(self):
        """Generate realistic sensor data dengan temporal patterns"""
        try:
            current_time = datetime.now()
            
            # Diurnal patterns (24-hour cycle)
            hour = current_time.hour
            minute = current_time.minute
            
            # Base values dengan diurnal variation
            if 6 <= hour <= 18:  # Daytime
                base_temp = 25 + 5 * np.sin(np.pi * (hour - 6) / 12)  # Peak at noon
                base_ph = 7.0 + 0.3 * np.sin(np.pi * (hour - 6) / 12)
            else:  # Nighttime
                base_temp = 20 - 2 * np.cos(np.pi * hour / 12)
                base_ph = 6.8 - 0.2 * np.cos(np.pi * hour / 12)
            
            # Add seasonal patterns (simplified)
            day_of_year = current_time.timetuple().tm_yday
            seasonal_temp_adj = 3 * np.sin(2 * np.pi * day_of_year / 365)
            seasonal_ph_adj = 0.1 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add random noise
            temp_noise = np.random.normal(0, 1.5)
            ph_noise = np.random.normal(0, 0.3)
            
            # Calculate final values
            suhu = np.clip(base_temp + seasonal_temp_adj + temp_noise, 10, 45)
            ph = np.clip(base_ph + seasonal_ph_adj + ph_noise, 4, 10)
            
            # Quality determination dengan improved logic
            quality_score = 0
            
            # pH score (optimal range: 6.5-7.5)
            if 6.5 <= ph <= 7.5:
                quality_score += 0.5
            elif 6.0 <= ph <= 8.0:
                quality_score += 0.3
            elif 5.5 <= ph <= 8.5:
                quality_score += 0.1
            
            # Temperature score (optimal range: 20-30°C)
            if 20 <= suhu <= 30:
                quality_score += 0.5
            elif 15 <= suhu <= 35:
                quality_score += 0.3
            elif 10 <= suhu <= 40:
                quality_score += 0.1
            
            # Add some randomness untuk realism
            quality_score += np.random.uniform(-0.1, 0.1)
            
            # Determine final quality
            if quality_score >= 0.7:
                kualitas = 'baik'
            elif quality_score >= 0.4:
                kualitas = 'sedang'
            else:
                kualitas = 'buruk'
            
            # Occasional anomalies (1% chance)
            if random.random() < 0.01:
                anomaly_type = random.choice(['ph_spike', 'temp_spike', 'ph_drop', 'temp_drop'])
                
                if anomaly_type == 'ph_spike':
                    ph = np.clip(ph + random.uniform(2, 4), 4, 14)
                    kualitas = 'buruk'
                elif anomaly_type == 'ph_drop':
                    ph = np.clip(ph - random.uniform(2, 3), 0, 14)
                    kualitas = 'buruk'
                elif anomaly_type == 'temp_spike':
                    suhu = np.clip(suhu + random.uniform(10, 20), 10, 60)
                    kualitas = 'buruk'
                elif anomaly_type == 'temp_drop':
                    suhu = np.clip(suhu - random.uniform(10, 15), -10, 45)
                    kualitas = 'buruk'
            
            return round(ph, 2), round(suhu, 2), kualitas
            
        except Exception as e:
            print(f"❌ Error generating sensor data: {e}")
            return 7.0, 25.0, 'buruk'
    
    def insert_data(self, ph, suhu, kualitas):
        """Insert data ke MySQL database"""
        try:
            insert_query = """
            INSERT INTO sensor_readings (ph, suhu, kualitas)
            VALUES (%s, %s, %s)
            """
            self.cursor.execute(insert_query, (ph, suhu, kualitas))
            self.connection.commit()
            
            self.stats['total_generated'] += 1
            self.stats['current_session'] += 1
            return True
                
        except mysql.connector.Error as err:
            print(f"❌ Insert error: {err}")
            self.stats['errors'] += 1
            return False
    
    def print_stats(self):
        """Print current statistics"""
        if self.stats['start_time']:
            runtime = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = self.stats['current_session'] / runtime if runtime > 0 else 0
            
            print(f"\n📊 SIMULATOR STATISTICS")
            print(f"{'='*40}")
            print(f"Runtime: {runtime/60:.1f} minutes")
            print(f"Records this session: {self.stats['current_session']}")
            print(f"Total generated: {self.stats['total_generated']}")
            print(f"Generation rate: {rate:.2f} records/sec")
            print(f"Errors: {self.stats['errors']}")
            print(f"Success rate: {((self.stats['current_session']-self.stats['errors'])/max(1,self.stats['current_session'])*100):.1f}%")
            print(f"{'='*40}")
    
    def run_simulation(self, interval=1.0, verbose=True):
        """Run simulation dengan specified interval"""
        
        if not self.connect_database():
            return False
            
        print("🚀 Starting Enhanced Sensor Data Simulation...")
        print(f"⏱️ Interval: {interval} seconds")
        print("📊 Generating realistic sensor data dengan temporal patterns")
        print("🔄 Press Ctrl+C to stop")
        print("="*60)
        
        self.stats['start_time'] = datetime.now()
        last_stats_time = time.time()
        
        try:
            while self.running:
                start_time = time.time()
                
                # Generate dan insert data
                ph, suhu, kualitas = self.generate_sensor_data()
                success = self.insert_data(ph, suhu, kualitas)
                
                if verbose and (self.stats['current_session'] % 10 == 0 or not success):
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    status = "✅" if success else "❌"
                    print(f"{status} [{timestamp}] pH={ph:5.2f}, Temp={suhu:5.1f}°C, Quality={kualitas:5s} | Total: {self.stats['current_session']}")
                
                # Print stats every 60 seconds
                if time.time() - last_stats_time >= 60:
                    self.print_stats()
                    last_stats_time = time.time()
                
                # Sleep untuk maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Simulation stopped by user")
        except Exception as e:
            print(f"\n❌ Simulation error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        self.print_stats()
        
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            
        print("✅ Simulation cleanup completed")
    
    def run_batch_simulation(self, count=100, batch_size=10):
        """Run batch simulation untuk testing"""
        
        if not self.connect_database():
            return False
            
        print(f"🧪 Running batch simulation: {count} records dalam batches of {batch_size}")
        
        self.stats['start_time'] = datetime.now()
        
        try:
            for batch in range(0, count, batch_size):
                batch_data = []
                
                for i in range(min(batch_size, count - batch)):
                    ph, suhu, kualitas = self.generate_sensor_data()
                    batch_data.append((ph, suhu, kualitas))
                
                # Insert batch
                for ph, suhu, kualitas in batch_data:
                    self.insert_data(ph, suhu, kualitas)
                
                print(f"✅ Batch {batch//batch_size + 1}: {len(batch_data)} records inserted")
                time.sleep(0.1)  # Small delay between batches
        
        except Exception as e:
            print(f"❌ Batch simulation error: {e}")
        finally:
            self.cleanup()

def main():
    """Main function"""
    simulator = SensorDataSimulator()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            simulator.run_batch_simulation(count)
        elif sys.argv[1] == '--fast':
            simulator.run_simulation(interval=0.5, verbose=True)
        elif sys.argv[1] == '--slow':
            simulator.run_simulation(interval=5.0, verbose=True)
        else:
            try:
                interval = float(sys.argv[1])
                simulator.run_simulation(interval=interval, verbose=True)
            except ValueError:
                print("❌ Invalid interval. Using default 1 second.")
                simulator.run_simulation(interval=1.0, verbose=True)
    else:
        # Default: run dengan 1 second interval
        simulator.run_simulation(interval=1.0, verbose=True)

if __name__ == "__main__":
    main()