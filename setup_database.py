#!/usr/bin/env python3
"""
Database Setup Script untuk Flutter Prediction Engine
Setup database sensor_data dengan semua table yang diperlukan
"""

import mysql.connector
import subprocess
import sys
import os

def check_mysql_service():
    """Check if MySQL service is running"""
    try:
        # For Windows (Laragon/XAMPP)
        result = subprocess.run(['sc', 'query', 'mysql'], capture_output=True, text=True)
        if 'RUNNING' in result.stdout:
            return True
        
        # Alternative check
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq mysqld.exe'], capture_output=True, text=True)
        if 'mysqld.exe' in result.stdout:
            return True
            
        return False
    except:
        return False

def test_connection(config):
    """Test database connection"""
    try:
        conn = mysql.connector.connect(**config)
        conn.close()
        return True
    except mysql.connector.Error as err:
        print(f"❌ Connection error: {err}")
        return False

def create_database(config):
    """Create sensor_data database if not exists"""
    try:
        # Connect without database
        temp_config = config.copy()
        temp_config.pop('database', None)
        
        conn = mysql.connector.connect(**temp_config)
        cursor = conn.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS sensor_data")
        print("✅ Database 'sensor_data' created/verified")
        
        cursor.close()
        conn.close()
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Database creation error: {err}")
        return False

def setup_database_from_sql():
    """Setup database using existing SQL file"""
    sql_file = os.path.join('database', 'sensor_data.sql')
    
    if not os.path.exists(sql_file):
        print(f"❌ SQL file not found: {sql_file}")
        return False
    
    try:
        print("📄 Importing database from sensor_data.sql...")
        
        # Use mysql command line tool
        cmd = [
            'mysql', 
            '-h', 'localhost',
            '-u', 'root',
            '-p',  # Will prompt for password (empty for Laragon)
            'sensor_data'
        ]
        
        with open(sql_file, 'r', encoding='utf-8') as f:
            result = subprocess.run(cmd, input=f.read(), text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ Database imported successfully")
            return True
        else:
            print(f"❌ Import error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Import exception: {e}")
        return False

def create_tables_manually(config):
    """Create tables manually if SQL import fails"""
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Create sensor_readings table (main table for our app)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                no int NOT NULL AUTO_INCREMENT,
                ph decimal(5,2) NOT NULL,
                suhu decimal(5,2) NOT NULL,
                kualitas enum('baik','sedang','buruk') NOT NULL,
                timestamp timestamp NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (no)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """)
        
        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id int NOT NULL AUTO_INCREMENT,
                timestamp datetime DEFAULT CURRENT_TIMESTAMP,
                model_type varchar(50) NOT NULL,
                ph float DEFAULT NULL,
                suhu float DEFAULT NULL,
                kualitas varchar(20) DEFAULT NULL,
                confidence float DEFAULT NULL,
                sensor_id int DEFAULT '1',
                PRIMARY KEY (id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
        """)
        
        # Insert sample data
        cursor.execute("""
            INSERT IGNORE INTO sensor_readings (no, ph, suhu, kualitas) VALUES
            (1, 7.20, 25.50, 'baik'),
            (2, 6.80, 23.00, 'sedang'),
            (3, 8.10, 27.20, 'baik'),
            (4, 5.50, 20.10, 'buruk'),
            (5, 7.80, 26.30, 'baik')
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Tables created successfully")
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Table creation error: {err}")
        return False

def verify_setup(config):
    """Verify database setup"""
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        print(f"📊 Found tables: {table_names}")
        
        # Check sensor_readings data
        if 'sensor_readings' in table_names:
            cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            count = cursor.fetchone()[0]
            print(f"📈 sensor_readings records: {count}")
        
        cursor.close()
        conn.close()
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ Verification error: {err}")
        return False

def main():
    print("🎯 Database Setup for Flutter Prediction Engine")
    print("=" * 50)
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',  # Empty for Laragon default
        'database': 'sensor_data'
    }
    
    # Check MySQL service
    print("🔍 Checking MySQL service...")
    if not check_mysql_service():
        print("❌ MySQL service not running!")
        print("💡 Please start MySQL service (Laragon/XAMPP)")
        return
    
    print("✅ MySQL service is running")
    
    # Create database
    print("\n📦 Setting up database...")
    if not create_database(db_config):
        return
    
    # Test connection with database
    print("🔗 Testing database connection...")
    if not test_connection(db_config):
        print("💡 Try these solutions:")
        print("   1. Check if MySQL is running in Laragon/XAMPP")
        print("   2. Verify username/password in config")
        print("   3. Check MySQL port (default: 3306)")
        return
    
    print("✅ Database connection successful")
    
    # Setup choice
    print("\n📄 Database setup options:")
    print("1. Import from sensor_data.sql (recommended)")
    print("2. Create minimal tables manually")
    
    choice = input("Choose option (1-2): ").strip()
    
    if choice == '1':
        # Try SQL import first
        if not setup_database_from_sql():
            print("⚠️ SQL import failed, trying manual setup...")
            create_tables_manually(db_config)
    else:
        create_tables_manually(db_config)
    
    # Verify setup
    print("\n🔍 Verifying database setup...")
    verify_setup(db_config)
    
    print("\n🎉 Database setup completed!")
    print("📝 Next steps:")
    print("   1. Run: python start_api.bat")
    print("   2. Run: start_flutter.bat")
    print("   3. Test Flutter app with predictions")

if __name__ == '__main__':
    main()
