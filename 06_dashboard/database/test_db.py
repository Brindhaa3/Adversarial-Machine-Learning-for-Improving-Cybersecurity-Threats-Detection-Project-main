import sqlite3

conn = sqlite3.connect("database/users.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM threats")
data = cursor.fetchall()

conn.close()

print("Threats in database:", data)
