"""Clear old data and reload from API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import text
from app.database import engine

with engine.connect() as conn:
    print("Clearing old data...")
    conn.execute(text("DELETE FROM predictions"))
    conn.execute(text("DELETE FROM anomalies"))
    conn.execute(text("DELETE FROM recommendations"))
    conn.execute(text("DELETE FROM pincode_metrics"))
    conn.execute(text("DELETE FROM raw_updates"))
    conn.commit()
    print("Old data cleared successfully!")
