import sqlite3

DB_PATH = 'schedules.db'

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute('DELETE FROM schedules;')
    conn.commit()
    print('All rows from the schedules table have been deleted. Table structure is preserved.')
except Exception as e:
    print(f'Error: {e}')
finally:
    conn.close()
