import sqlite3
import csv
from pathlib import Path

# Configuration
DB_NAME = "taxi_demand.db"
TABLE_NAME = "demand_data"
CSV_FILE = "taxis_dataset_with_features.csv"  # Input CSV file path
OUTPUT_CSV = "taxis_dataset_final.csv"  # Output CSV file path

def create_db_and_load_data():
    """Create database and load CSV data"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Drop table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    
    # Create table
    cursor.execute(f"""
    CREATE TABLE {TABLE_NAME} (
        year INTEGER,
        month INTEGER,
        day INTEGER,
        hour INTEGER,
        pickup_longitude REAL,
        pickup_latitude REAL,
        demand INTEGER,
        is_holiday INTEGER,
        temperature REAL,
        precipitation REAL
    )
    """)
    
    # Load data from CSV
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cursor.execute(f"""
            INSERT INTO {TABLE_NAME} VALUES (
                :year, :month, :day, :hour,
                :pickup_longitude, :pickup_latitude,
                :demand, :is_holiday,
                :temperature, :precipitation
            )
            """, row)
    
    conn.commit()
    print(f"Data loaded from {CSV_FILE} to {TABLE_NAME} table")
    return conn

def update_holiday_flag(conn):
    """Update is_holiday flag for specific dates"""
    cursor = conn.cursor()
    
    # First reset all holidays to 0
    cursor.execute(f"UPDATE {TABLE_NAME} SET is_holiday = 0")
    
    # Set holiday flag for 2016-01-01 and 2016-01-19
    cursor.execute(f"""
    UPDATE {TABLE_NAME} 
    SET is_holiday = 1 
    WHERE (year = 2016 AND month = 1 AND day = 1)
       OR (year = 2016 AND month = 1 AND day = 19)
    """)
    
    conn.commit()
    print("Updated holiday flags for 2016-01-01 and 2016-01-19")

def export_to_csv(conn):
    """Export data to CSV without precipitation column"""
    cursor = conn.cursor()
    
    # Get data without precipitation column
    cursor.execute(f"""
    SELECT 
        year, month, day, hour,
        pickup_longitude, pickup_latitude,
        demand, is_holiday, temperature
    FROM {TABLE_NAME}
    """)
    
    # Get column names
    columns = [description[0] for description in cursor.description]
    
    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)  # Write header
        writer.writerows(cursor.fetchall())
    
    print(f"\nData exported to {OUTPUT_CSV} without precipitation column")

def main():
    # Check if CSV file exists
    if not Path(CSV_FILE).exists():
        print(f"Error: CSV file not found at {CSV_FILE}")
        return
    
    # Create DB and load data
    conn = create_db_and_load_data()
    
    # Update holiday flag
    update_holiday_flag(conn)
    
    # Export to CSV without precipitation column
    export_to_csv(conn)
    
    # Close connection
    conn.close()
    print("\nAll operations completed successfully")

if __name__ == "__main__":
    main()