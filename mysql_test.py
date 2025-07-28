#!/usr/bin/env python3
"""
Test MySQL connection and setup
"""

import mysql.connector
from mysql.connector import Error

def test_mysql_connection():
    """Test MySQL connection"""
    try:
        # Database configuration
        config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',  # Sesuaikan dengan password MySQL Anda
            'database': 'sensor_data'
        }
        
        print("🔍 Testing MySQL connection...")
        print(f"   Host: {config['host']}")
        print(f"   User: {config['user']}")
        print(f"   Database: {config['database']}")
        
        # Test connection
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        print("✅ MySQL connection successful!")
        
        # Test database exists
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()
        print(f"✅ Using database: {db_name[0]}")
        
        # Create sensor table
        # Drop existing table first untuk ensure clean structure
        cursor.execute("DROP TABLE IF EXISTS sensor_readings")
        
        create_table_query = """
        CREATE TABLE sensor_readings (
            no INT AUTO_INCREMENT PRIMARY KEY,
            ph DECIMAL(5,2) NOT NULL,
            suhu DECIMAL(5,2) NOT NULL,
            kualitas ENUM('baik', 'buruk') NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp)
        )
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print("✅ Table 'sensor_readings' created/verified")
        
        # Test insert
        insert_query = """
        INSERT INTO sensor_readings (ph, suhu, kualitas)
        VALUES (%s, %s, %s)
        """
        test_data = (7.2, 25.5, 'baik')
        cursor.execute(insert_query, test_data)
        connection.commit()
        print("✅ Test data inserted successfully")
        
        # Test select
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        count = cursor.fetchone()
        print(f"✅ Total records in database: {count[0]}")
        
        # Cleanup
        cursor.close()
        connection.close()
        print("✅ Connection closed")
        
        return True
        
    except Error as e:
        print(f"❌ MySQL Error: {e}")
        return False
    except Exception as e:
        print(f"❌ General Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 MySQL Connection Test")
    print("=" * 40)
    
    success = test_mysql_connection()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 MySQL setup successful! Ready to run simulator.")
    else:
        print("❌ MySQL setup failed. Please check:")
        print("   1. MySQL server is running (check Laragon)")
        print("   2. Database 'sensor_data' exists")
        print("   3. Username/password is correct")
        print("   4. MySQL service is accessible")
