import sqlite3

# Connect to SQLite database (creates if it doesn't exist)
conn = sqlite3.connect("database/users.db")
cursor = conn.cursor()

# Create threats table
cursor.execute("""
CREATE TABLE IF NOT EXISTS threats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    attack_name TEXT NOT NULL,
    severity INTEGER NOT NULL
);
""")

# Insert some sample threats
cursor.executemany("""
INSERT INTO threats (timestamp, attack_name, severity) VALUES (?, ?, ?)
""", [
    ('2025-03-21 12:00:00', 'Brute Force', 4),
    ('2025-03-21 12:05:00', 'Phishing', 2),
    ('2025-03-21 12:10:00', 'Ransomware', 5),
    ('2025-03-21 12:15:00', 'DDoS', 3),
    ('2025-03-21 12:20:00', 'SQL Injection', 4)
])

# Commit and close
conn.commit()
conn.close()

print(" Database setup complete! ")
 
 
