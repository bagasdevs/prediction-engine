import mysql.connector
from datetime import datetime

try:
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='sensor_data'
    )
    
    cursor = db.cursor()
    
    # Count total predictions
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]
    print(f'📊 Total predictions: {total}')
    
    # Get latest predictions
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10')
    results = cursor.fetchall()
    
    print('\n🔍 Latest predictions:')
    for row in results:
        print(f"ID: {row[0]}, Time: {row[1]}, Model: {row[2]}, pH: {row[3]:.2f}, Suhu: {row[4]:.2f}, Kualitas: {row[5]}, Confidence: {row[6]:.2f}")
    
    # Check if new predictions are being generated (within last minute)
    cursor.execute("""
        SELECT COUNT(*) FROM predictions 
        WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 1 MINUTE)
    """)
    recent = cursor.fetchone()[0]
    print(f'\n⏰ Predictions in last minute: {recent}')
    
    if recent > 0:
        print('✅ ML Engine is generating real-time predictions!')
    else:
        print('⚠️ No recent predictions - checking if system is running...')
    
    db.close()
    
except Exception as e:
    print(f'❌ Database error: {e}')
