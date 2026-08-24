import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_path="output/car_system_logs.db"):
        os.makedirs("output", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()
        
        # GÜNCELLEME: Sürekli commit atmamak için sayaç
        self.log_counter = 0

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                frame_no INTEGER,
                detector TEXT,
                tracker TEXT,
                distance_type TEXT,
                distance REAL,
                iou REAL,
                fps REAL
            )
        ''')
        self.conn.commit()

    def log_data(self, frame_no, detector, tracker, dist_type, distance, iou, fps):
        self.cursor.execute('''
            INSERT INTO logs (frame_no, detector, tracker, distance_type, distance, iou, fps)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (frame_no, detector, tracker, dist_type, distance, iou, fps))
        
        # GÜNCELLEME: SQLite Darboğazını (Lock) çözmek için 100 logda bir commit.
        self.log_counter += 1
        if self.log_counter % 100 == 0:
            self.conn.commit()

    def get_latest_logs(self, limit=10):
        self.cursor.execute('''
            SELECT timestamp, frame_no, detector, tracker, distance_type, distance, iou, fps 
            FROM logs ORDER BY id DESC LIMIT ?
        ''', (limit,))
        return self.cursor.fetchall()

    def get_all_logs(self):
        self.cursor.execute('''
            SELECT timestamp, frame_no, detector, tracker, distance_type, distance, iou, fps 
            FROM logs ORDER BY id DESC
        ''')
        return self.cursor.fetchall()

    def close(self):
        self.conn.commit() # Kapanırken son kalanları da yazsın
        self.conn.close()