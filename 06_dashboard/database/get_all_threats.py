import sqlite3
import os

DB_PATH = "database/users.db"

def initialize_db():
    """Create database and table if not exists."""
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

def get_all_threats():
    """Fetch all threats from the database, sorted by timestamp (latest first)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch all threats, ordered by timestamp (newest first)
    cursor.execute("SELECT * FROM threats ORDER BY timestamp DESC")
    threats = cursor.fetchall()  # Get all rows

    conn.close()

    # Format the threats into a list of dictionaries for easier use in the frontend
    formatted_threats = []
    for threat in threats:
        formatted_threats.append({
            "timestamp": threat[1],  # timestamp is in column 1
            "attack_name": threat[2],  # attack_name is in column 2
            "severity": threat[3]  # severity is in column 3
        })

    return formatted_threats

def add_threat(timestamp, attack_name, severity):
    """Insert a new threat into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Insert the new threat into the database
    cursor.execute("INSERT INTO threats (timestamp, attack_name, severity) VALUES (?, ?, ?)",
                   (timestamp, attack_name, severity))

    conn.commit()
    conn.close()

    print(f"Added new threat: {attack_name} (Severity {severity})")

# To initialize the database and create the table (if not done yet)
# initialize_db()
