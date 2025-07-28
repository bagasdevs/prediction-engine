import mysql.connector
import random
import time
from datetime import datetime
import signal
import sys

class SensorDataSimulator:
    def __init__(self):
        # Konfigurasi database
        self.db_config = {
            'host': 'localhost',
            'user': 'root',  
            'password': '',  
            'database': 'sensor_data'
        }
        
        self.connection = None
        self.cursor = None
        self.running = True
        self.data_counter = 1
        
        # Setup signal handler untuk stop yang aman
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def connect_to_database(self):
        """Koneksi ke database MySQL"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor()
            print("✅ Berhasil terhubung ke database MySQL")
            return True
        except mysql.connector.Error as err:
            print(f"❌ Error koneksi database: {err}")
            return False
    
    def generate_sensor_data(self):
        """Generate data sensor secara acak"""
        # Generate pH (0-14, dengan distribusi normal sekitar 7)
        ph = round(random.uniform(4.0, 10.0), 2)
        
        # Generate suhu (15-40°C)
        suhu = round(random.uniform(15.0, 40.0), 2)
        
        # Tentukan kualitas berdasarkan pH dan suhu
        # pH ideal: 6.5-7.5, Suhu ideal: 20-30°C
        if (6.5 <= ph <= 7.5) and (20 <= suhu <= 30):
            # 80% kemungkinan baik jika dalam range ideal
            kualitas = 'baik' if random.random() < 0.8 else 'buruk'
        elif (6.0 <= ph <= 8.0) and (18 <= suhu <= 35):
            # 60% kemungkinan baik jika dalam range acceptable
            kualitas = 'baik' if random.random() < 0.6 else 'buruk'
        else:
            # 30% kemungkinan baik jika diluar range
            kualitas = 'baik' if random.random() < 0.3 else 'buruk'
        
        return {
            'no': self.data_counter,
            'ph': ph,
            'suhu': suhu,
            'kualitas': kualitas
        }
    
    def insert_data(self, data):
        """Insert data ke database"""
        try:
            query = """
            INSERT INTO sensor_readings (no, ph, suhu, kualitas) 
            VALUES (%(no)s, %(ph)s, %(suhu)s, %(kualitas)s)
            """
            
            self.cursor.execute(query, data)
            self.connection.commit()
            
            # Tampilkan data yang diinsert
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Data #{data['no']}: pH={data['ph']}, Suhu={data['suhu']}°C, Kualitas={data['kualitas']}")
            
            return True
            
        except mysql.connector.Error as err:
            print(f"❌ Error insert data: {err}")
            return False
    
    def get_latest_data_count(self):
        """Ambil jumlah data terakhir untuk melanjutkan counter"""
        try:
            query = "SELECT COUNT(*) as total FROM sensor_readings"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return result[0] + 1 if result else 1
        except mysql.connector.Error as err:
            print(f"❌ Error getting data count: {err}")
            return 1
    
    def show_statistics(self):
        """Tampilkan statistik data"""
        try:
            query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN kualitas = 'baik' THEN 1 ELSE 0 END) as baik,
                SUM(CASE WHEN kualitas = 'buruk' THEN 1 ELSE 0 END) as buruk,
                AVG(ph) as avg_ph,
                AVG(suhu) as avg_suhu
            FROM sensor_readings
            """
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result:
                total, baik, buruk, avg_ph, avg_suhu = result
                print(f"\n📊 Statistik Data:")
                print(f"   Total data: {total}")
                print(f"   Kualitas baik: {baik} ({baik/total*100:.1f}%)")
                print(f"   Kualitas buruk: {buruk} ({buruk/total*100:.1f}%)")
                print(f"   Rata-rata pH: {avg_ph:.2f}")
                print(f"   Rata-rata suhu: {avg_suhu:.2f}°C")
                print("-" * 40)
                
        except mysql.connector.Error as err:
            print(f"❌ Error getting statistics: {err}")
    
    def signal_handler(self, signum, frame):
        """Handler untuk Ctrl+C"""
        print("\n🛑 Menghentikan simulator...")
        self.running = False
    
    def cleanup(self):
        """Bersihkan koneksi database"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("✅ Koneksi database ditutup")
    
    def run(self):
        """Jalankan simulator"""
        print("🚀 Memulai Sensor Data Simulator")
        print("📝 Tekan Ctrl+C untuk menghentikan")
        print("-" * 40)
        
        # Koneksi ke database
        if not self.connect_to_database():
            return
        
        # Set counter berdasarkan data yang sudah ada
        self.data_counter = self.get_latest_data_count()
        print(f"📊 Memulai dari data ke-{self.data_counter}")
        print("-" * 40)
        
        # Loop utama
        stats_counter = 0
        try:
            while self.running:
                # Generate dan insert data
                sensor_data = self.generate_sensor_data()
                
                if self.insert_data(sensor_data):
                    self.data_counter += 1
                    stats_counter += 1
                    
                    # Tampilkan statistik setiap 100 data
                    if stats_counter >= 100:
                        self.show_statistics()
                        stats_counter = 0
                
                # Tunggu 1 detik
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.show_statistics()
            self.cleanup()
            print("👋 Simulator dihentikan")

if __name__ == "__main__":
    simulator = SensorDataSimulator()
    simulator.run()