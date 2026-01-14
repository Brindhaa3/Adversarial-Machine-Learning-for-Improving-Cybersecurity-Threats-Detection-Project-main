import sqlite3

conn = sqlite3.connect("06_dashboard/database/users.db")  # Adjust if needed
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

if users:
    print(" Users Found:", users)
else:
    print(" No users found in the database!")

conn.close()
