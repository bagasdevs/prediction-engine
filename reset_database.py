#!/usr/bin/env python3
"""
Database Reset - Clean setup untuk fresh start
"""

import mysql.connector
from mysql.connector import Error

def reset_database():
    """Reset database dengan clean setup"""
    try:
        # Connect to MySQL (without specific database)
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=''
        )
        cursor = connection.cursor()
        
        print("🧹 Database Reset Starting...")
        
        # Drop database jika exists
        cursor.execute("DROP DATABASE IF EXISTS sensor_data")
        print("✅ Old database dropped")
        
        # Create fresh database
        cursor.execute("CREATE DATABASE sensor_data")
        print("✅ Fresh database created")
        
        # Switch to new database
        cursor.execute("USE sensor_data")
        
        # Create sensor_readings table
        create_sensor_table = """
        CREATE TABLE sensor_readings (
            no INT AUTO_INCREMENT PRIMARY KEY,
            ph DECIMAL(5,2) NOT NULL,
            suhu DECIMAL(5,2) NOT NULL,
            kualitas ENUM('baik', '', 'buruk') NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_timestamp (timestamp)
        )
        """
        cursor.execute(create_sensor_table)
        print("✅ sensor_readings table created")
        
        # Create predictions table untuk AI results
        create_predictions_table = """
        CREATE TABLE ai_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ph_predicted DECIMAL(5,2),
            suhu_predicted DECIMAL(5,2), 
            kualitas_predicted ENUM('baik', 'buruk'),
            confidence DECIMAL(5,3),
            model_used VARCHAR(50),
            INDEX idx_timestamp (timestamp)
        )
        """
        cursor.execute(create_predictions_table)
        print("✅ ai_predictions table created")
        
        # Test insert
        test_insert = """
        INSERT INTO sensor_readings (ph, suhu, kualitas)
        VALUES (7.2, 25.5, 'baik')
        """
        cursor.execute(test_insert)
        connection.commit()
        print("✅ Test data inserted")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        count = cursor.fetchone()[0]
        print(f"✅ Table verified: {count} records")
        
        connection.close()
        print("🎉 Database reset completed successfully!")
        return True
        
    except Error as e:
        print(f"❌ Database reset error: {e}")
        return False

if __name__ == "__main__":
    reset_database()
