import sqlite3
import time
from flask import current_app

DB_PATH = "database/users.db"

def add_threat(timestamp, attack_name, severity):
    """Insert a new threat into the database and trigger live update"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO threats (timestamp, attack_name, severity) VALUES (?, ?, ?)",
                   (timestamp, attack_name, severity))
    conn.commit()
    conn.close()
    
    # Access socketio from the current Flask app context
    socketio = current_app.extensions['socketio']
    socketio.emit("update_data", get_all_threats())  # Emit updated threats to clients
    print(f"Added new threat: {attack_name} (Severity {severity})")

def get_all_threats():
    """Fetch all threats from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM threats ORDER BY timestamp DESC")
    threats = cursor.fetchall()
    conn.close()
    return [{"ID": row[0], "Timestamp": row[1], "Name": row[2], "Severity": row[3]} for row in threats]
