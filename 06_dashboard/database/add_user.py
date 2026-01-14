import sqlite3
import hashlib

def get_db_connection():
    """Function to connect to the SQLite database."""
    conn = sqlite3.connect("06_dashboard/database/users.db")
    conn.row_factory = sqlite3.Row
    return conn

# Define user details
username = "admin"
password = "admin123"
role = "admin"  #  ADD ROLE HERE

# Hash the password
hashed_password = hashlib.sha256(password.encode()).hexdigest()

# Connect to the database
conn = get_db_connection()
cursor = conn.cursor()

# Ensure the table has a role column
cursor.execute("PRAGMA table_info(users);")
columns = [column[1] for column in cursor.fetchall()]
if 'role' not in columns:
    cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    conn.commit()

# Check if the username already exists
cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
existing_user = cursor.fetchone()

if existing_user:
    print(f" User '{username}' already exists. Skipping insertion.")
else:
    # Insert user data only if the username is not taken
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                   (username, hashed_password, role))
    conn.commit()
    print(f" User '{username}' added with role '{role}' successfully!")

# Close the connection
conn.close()
