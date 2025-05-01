from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sqlite3

app = FastAPI()

@app.get("/api/schedules")
def get_schedules():
    conn = sqlite3.connect("schedules.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedules")
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    schedules = [dict(zip(col_names, row)) for row in rows]
    conn.close()
    return JSONResponse(content=schedules)
