import sqlite3
import time
import random

DB_PATH = "database/users.db"

ATTACKS = [
    ("DDoS Attack", 4),
    ("SQL Injection", 3),
    ("Brute Force Attack", 2),
    ("Man-in-the-Middle", 4),
    ("Phishing Attempt", 1),
    ("Ransomware", 5),
    ("Zero-Day Exploit", 5),
    ("XSS Attack", 3),
    ("Malware Infection", 4),
]

def add_threat(timestamp, attack_name, severity):
    """Insert a new threat into the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO threats (timestamp, attack_name, severity) VALUES (?, ?, ?)",
                   (timestamp, attack_name, severity))
    conn.commit()
    conn.close()
    print(f"Added new threat: {attack_name} (Severity {severity})")

def generate_threats():
    """Continuously add new threats to simulate attacks"""
    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        attack_name, severity = random.choice(ATTACKS)
        add_threat(timestamp, attack_name, severity)
        time.sleep(random.randint(5, 15))  # New threat every 5-15 seconds

if __name__ == "__main__":
    print("Starting live attack simulation...")
    generate_threats()
