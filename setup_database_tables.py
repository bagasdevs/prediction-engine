#!/usr/bin/env python3
"""
Database Table Setup untuk ML Engine
Membuat tabel predictions yang diperlukan
"""

import mysql.connector
from mysql.connector import Error

def create_predictions_table():
    """Create predictions table with correct structure"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        cursor = connection.cursor()
        
        # Drop existing predictions table if exists
        cursor.execute("DROP TABLE IF EXISTS predictions")
        print("🗑️ Dropped existing predictions table")
        
        # Create new predictions table
        create_table_query = """
        CREATE TABLE predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_type VARCHAR(50) NOT NULL,
            ph FLOAT,
            suhu FLOAT,
            kualitas VARCHAR(20),
            confidence FLOAT,
            sensor_id INT DEFAULT 1,
            INDEX idx_timestamp (timestamp),
            INDEX idx_model_type (model_type)
        )
        """
        
        cursor.execute(create_table_query)
        print("✅ Created predictions table with correct structure")
        
        # Insert some sample predictions for testing
        sample_predictions = [
            ('cnn_prediction', 7.2, 25.5, 'baik', 0.85),
            ('lstm_prediction', 7.1, 25.3, 'baik', 0.82),
            ('hybrid_prediction', 7.15, 25.4, 'baik', 0.88),
            ('cnn_prediction', 6.8, 26.1, 'buruk', 0.79),
            ('lstm_prediction', 6.9, 26.0, 'buruk', 0.81),
            ('hybrid_prediction', 6.85, 26.05, 'buruk', 0.83)
        ]
        
        insert_query = """
        INSERT INTO predictions (model_type, ph, suhu, kualitas, confidence)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.executemany(insert_query, sample_predictions)
        print(f"✅ Inserted {len(sample_predictions)} sample predictions")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("🎉 Database setup completed successfully!")
        return True
        
    except Error as e:
        print(f"❌ Database error: {e}")
        return False

def verify_tables():
    """Verify all required tables exist"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='sensor_data',
            port=3306
        )
        
        cursor = connection.cursor()
        
        # Check sensor_readings table
        cursor.execute("SHOW TABLES LIKE 'sensor_readings'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM sensor_readings")
            sensor_count = cursor.fetchone()[0]
            print(f"✅ sensor_readings table: {sensor_count} records")
        else:
            print("❌ sensor_readings table missing")
        
        # Check predictions table
        cursor.execute("SHOW TABLES LIKE 'predictions'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) FROM predictions")
            pred_count = cursor.fetchone()[0]
            print(f"✅ predictions table: {pred_count} records")
        else:
            print("❌ predictions table missing")
        
        cursor.close()
        connection.close()
        
        return True
        
    except Error as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    print("🔧 DATABASE TABLE SETUP")
    print("=" * 40)
    
    print("\n1️⃣ Creating predictions table...")
    if create_predictions_table():
        print("\n2️⃣ Verifying tables...")
        verify_tables()
        
        print("\n" + "=" * 40)
        print("🎉 SETUP COMPLETED!")
        print("✅ Predictions table ready")
        print("✅ Sample data inserted")
        print("🚀 Dashboard should now work without warnings")
    else:
        print("\n❌ Setup failed")
        print("💡 Check MySQL connection and try again")

if __name__ == "__main__":
    main()
