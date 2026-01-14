import sqlite3
import os

DB_PATH = "database/users.db"

def initialize_db():
    """Create database and table if not exists."""
    try:
        if not os.path.exists("database"):
            os.makedirs("database")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create threats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                attack_name TEXT NOT NULL,
                severity INTEGER NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except sqlite3.DatabaseError as e:
        print("Database error:", e)
    except Exception as e:
        print("General error:", e)

def get_all_threats():
    """Fetch all threats from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM threats ORDER BY timestamp DESC")
    threats = cursor.fetchall()
    conn.close()
    return [{"id": t[0], "timestamp": t[1], "attack_name": t[2], "severity": t[3]} for t in threats]

def add_threat(timestamp, attack_name, severity):
    """Insert a new threat into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO threats (timestamp, attack_name, severity) VALUES (?, ?, ?)",
                   (timestamp, attack_name, severity))
    conn.commit()
    conn.close()
    print(f"Added new threat: {attack_name} (Severity {severity})")
