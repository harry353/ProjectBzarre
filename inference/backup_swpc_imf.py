import sqlite3
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DB = PROJECT_ROOT / "inference" / "backup_imf_swpc.db"
TARGET_DB = PROJECT_ROOT / "inference" / "space_weather_last_6m.db"
SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"

def init_db():
    with sqlite3.connect(BACKUP_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imf_swpc (
                time_tag TEXT PRIMARY KEY,
                bt REAL,
                bx REAL,
                by REAL,
                bz REAL
            )
        """)
        conn.commit()

def download_swpc_mag():
    try:
        resp = requests.get(SWPC_MAG_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"[ERROR] Failed to download SWPC MAG data: {exc}")
        return None

    if not isinstance(payload, list) or len(payload) < 2:
        print("[WARN] Unexpected SWPC MAG payload format")
        return None

    header = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=header)
    
    # Standardize columns
    mapping = {
        "time_tag": "time_tag",
        "bt": "bt",
        "bx_gsm": "bx",
        "by_gsm": "by",
        "bz_gsm": "bz"
    }
    
    # SWPC columns are sometimes lowercase or slightly different
    cols_lower = {c.lower(): c for c in df.columns}
    rename_map = {}
    for swpc_key, internal_key in mapping.items():
        if swpc_key in cols_lower:
            rename_map[cols_lower[swpc_key]] = internal_key
            
    df = df.rename(columns=rename_map)
    
    # Convert types
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    for col in ("bt", "bx", "by", "bz"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    df = df.dropna(subset=["time_tag"])
    df["time_tag"] = df["time_tag"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    
    return df[["time_tag", "bt", "bx", "by", "bz"]]

def backup_data(df):
    if df is None or df.empty:
        return
    
    with sqlite3.connect(BACKUP_DB) as conn:
        cursor = conn.cursor()
        data_to_insert = df.values.tolist()
        cursor.executemany("""
            INSERT OR IGNORE INTO imf_swpc (time_tag, bt, bx, by, bz)
            VALUES (?, ?, ?, ?, ?)
        """, data_to_insert)
        added = cursor.rowcount
        conn.commit()
    print(f"[OK] Added {added} new records to {BACKUP_DB.name}")

def prune_old_data():
    # 3 months is roughly 90 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S+00:00")
    with sqlite3.connect(BACKUP_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM imf_swpc WHERE time_tag < ?", (cutoff,))
        removed = cursor.rowcount
        conn.commit()
    if removed:
        print(f"[CLEANUP] Pruned {removed} records older than 90 days from backup_imf_swpc.db")

def main():
    print(f"--- SWPC IMF Backup Started {datetime.now().isoformat()} ---")
    init_db()
    df = download_swpc_mag()
    backup_data(df)
    prune_old_data()
    print("--- Backup Complete ---")

if __name__ == "__main__":
    main()
