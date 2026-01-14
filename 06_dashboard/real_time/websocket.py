import json
import time
import sqlite3
from flask import Flask
from flask_socketio import SocketIO

DB_PATH = "database/users.db"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def get_all_threats():
    """Fetch all threats from the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM threats ORDER BY timestamp DESC")
    threats = cursor.fetchall()
    conn.close()
    return [{"id": t[0], "timestamp": t[1], "attack_name": t[2], "severity": t[3]} for t in threats]

@socketio.on("connect")
def handle_connect():
    print("Client connected")
    send_threats()

def send_threats():
    """Send latest threats to the frontend"""
    threats = get_all_threats()
    print("Sending threats:", threats)
    socketio.emit("update_threats", threats)

def monitor_database():
    """Continuously checks for new threats and updates clients"""
    last_count = 0
    while True:
        time.sleep(5)  # Check every 5 seconds
        threats = get_all_threats()
        if len(threats) > last_count:
            last_count = len(threats)
            send_threats()

if __name__ == "__main__":
    socketio.start_background_task(monitor_database)
    socketio.run(app, debug=True, port=5000)
