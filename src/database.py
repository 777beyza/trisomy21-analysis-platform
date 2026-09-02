import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "data/trisomy21.db"

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create caregiver logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS caregiver_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_date TEXT UNIQUE,
        apathy INTEGER,
        stubbornness INTEGER,
        memory INTEGER,
        total_score INTEGER
    )
    ''')
    
    conn.commit()
    conn.close()

def save_caregiver_log(log_date, apathy, stubbornness, memory):
    """Saves a daily log to the database. Overwrites if the same date exists."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    total_score = apathy + stubbornness + memory
    
    # Use REPLACE to update if log_date already exists (since it's UNIQUE)
    cursor.execute('''
    INSERT OR REPLACE INTO caregiver_logs (log_date, apathy, stubbornness, memory, total_score)
    VALUES (?, ?, ?, ?, ?)
    ''', (log_date, apathy, stubbornness, memory, total_score))
    
    conn.commit()
    conn.close()

def get_caregiver_logs():
    """Retrieves all logs ordered by date as a Pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    
    # Read directly into pandas DataFrame
    query = "SELECT * FROM caregiver_logs ORDER BY log_date ASC"
    df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    # If the dataframe is not empty, rename columns for better UI display
    if not df.empty:
        df = df.rename(columns={
            'log_date': 'Gün',
            'apathy': 'Apati Skoru',
            'stubbornness': 'İnatçılık Skoru',
            'memory': 'Hafıza Skoru',
            'total_score': 'Score'
        })
        # Convert string dates to datetime objects for plotly
        df['Gün'] = pd.to_datetime(df['Gün'])
        
    return df
