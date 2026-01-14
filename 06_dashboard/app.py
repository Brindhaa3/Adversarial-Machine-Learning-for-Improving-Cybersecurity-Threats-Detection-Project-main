import json
import time
import sqlite3
from flask import Flask, render_template, redirect, url_for, request, session
from flask_socketio import SocketIO
from database import initialize_db, add_threat, get_all_threats
from random import choice, randint
from datetime import datetime

DB_PATH = "database/users.db"

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management
socketio = SocketIO(app, cors_allowed_origins="*")

# Sample attack names for automatic generation
attack_names = ["DDoS Attack", "SQL Injection", "Brute Force Attack", "Phishing Attempt", "Man-in-the-Middle"]

# Dummy username and password
DUMMY_USERNAME = "admin"
DUMMY_PASSWORD = "dummy123"

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
        # Automatically add a new threat every 5 seconds
        attack_name = choice(attack_names)  # Choose a random attack name
        severity = randint(1, 5)  # Random severity between 1 and 5
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current time as timestamp

        # Add the new threat to the database
        add_threat(timestamp, attack_name, severity)

        threats = get_all_threats()
        if len(threats) > last_count:
            last_count = len(threats)
            send_threats()

@app.route('/')
def index():
    """Serve the dashboard page if logged in, otherwise show login page"""
    if 'username' in session:
        return render_template("dashboard.html")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle the login functionality"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username and password match the dummy credentials
        if username == DUMMY_USERNAME and password == DUMMY_PASSWORD:
            session['username'] = username  # Store username in session
            return redirect(url_for('index'))  # Redirect to the dashboard
        else:
            return "Invalid credentials, please try again", 401  # Unauthorized

    return render_template("login.html")

@app.route('/logout')
def logout():
    """Handle logout and clear session"""
    session.pop('username', None)  # Remove username from session
    return redirect(url_for('login'))  # Redirect to login page

if __name__ == "__main__":
    initialize_db()  # Initialize the database once
    socketio.start_background_task(monitor_database)
    socketio.run(app, debug=True, port=5000)
