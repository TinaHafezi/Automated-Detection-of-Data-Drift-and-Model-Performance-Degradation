import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np

class MetricsStore:

    def __init__(self, db_path="metrics.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            timestamp TEXT,
            metric_name TEXT,
            value REAL
        )
        """)
        conn.commit()
        conn.close()

    def save_metrics(self, metrics_dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for k, v in metrics_dict.items():
            c.execute("INSERT INTO metrics VALUES (?, ?, ?)",
                      (datetime.now().isoformat(), k, float(v)))
        conn.commit()
        conn.close()

    def load_history(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM metrics", conn)
        conn.close()
        return df
    
    def get_historical_stats(self, metric_name, window=30):
        conn = sqlite3.connect("metrics.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT value FROM metrics
            WHERE metric_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (metric_name, window))

        values = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(values) < 5:
            return None, None  # not enough history

        return np.mean(values), np.std(values)
