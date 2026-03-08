from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKUP_DB = PROJECT_ROOT / "inference" / "backup_imf_swpc.db"
TARGET_DB = PROJECT_ROOT / "inference" / "space_weather_last_6m.db"

# Remove all records from the target table that were previously inserted as SWPC backup data
def delete_swpc_backup():
    if not TARGET_DB.exists():
        return
    with sqlite3.connect(TARGET_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dscovr_m1m WHERE source_type = 'swpc_backup'")
        removed = cursor.rowcount
        conn.commit()
    if removed:
        print(f"[CLEANUP] Removed {removed} 'swpc_backup' records from {TARGET_DB.name}")

# Main synchronization logic: Load backup data -> Filter duplicates -> Insert -> Globally sort target table
def main() -> None:
    print(f"--- SWPC IMF Backup Insertion Started {datetime.now(timezone.utc).isoformat()} ---")
    
    # 1. Clear out ALL existing swpc_backup data to ensure a clean sync
    delete_swpc_backup()

    if not BACKUP_DB.exists():
        print(f"[WARN] {BACKUP_DB} not found. Skipping.")
        return

    if not TARGET_DB.exists():
        print(f"[WARN] {TARGET_DB} not found. Skipping.")
        return

    # Load all records from the backup database
    with sqlite3.connect(BACKUP_DB) as b_conn:
        b_df = pd.read_sql_query("SELECT * FROM imf_swpc", b_conn)

    if b_df.empty:
        print("[INFO] No data in backup_imf_swpc.db.")
        return

    # Standardize time_tag format to match DSCOVR's internal storage format: %Y-%m-%d %H:%M:%S+00:00
    b_df["time_tag"] = pd.to_datetime(b_df["time_tag"], errors="coerce", utc=True)
    b_df = b_df.dropna(subset=["time_tag"])
    b_df["time_tag"] = b_df["time_tag"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    b_df["source_type"] = "swpc_backup"

    # Identify existing timestamps in the target table to avoid redundant insertions
    with sqlite3.connect(TARGET_DB) as t_conn:
        t_df = pd.read_sql_query("SELECT time_tag FROM dscovr_m1m", t_conn)
        if not t_df.empty:
            t_df["time_tag"] = pd.to_datetime(t_df["time_tag"], errors="coerce", utc=True)
            t_df = t_df.dropna(subset=["time_tag"])
            existing_tags = set(t_df["time_tag"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00"))
        else:
            existing_tags = set()

    # Filter out any backup records that are already present in the target table
    mask = ~b_df["time_tag"].isin(existing_tags)
    new_data = b_df[mask]

    if new_data.empty:
        print("[INFO] No new SWPC IMF data to insert.")
    else:

        # Sort values chronologically before appending to the database
        new_data = new_data.sort_values(by="time_tag")

        # Append new records into the primary dscovr_m1m table
        with sqlite3.connect(TARGET_DB) as t_conn:
            new_data.to_sql("dscovr_m1m", t_conn, if_exists="append", index=False)
            t_conn.commit()

        print(f"[OK] Added {len(new_data)} SWPC IMF backup records into dscovr_m1m.")

    # Re-sort the ENTIRE table to ensure chronological data flow after backup injection
    with sqlite3.connect(TARGET_DB) as t_conn:
        t_df_full = pd.read_sql_query("SELECT * FROM dscovr_m1m", t_conn)
        if not t_df_full.empty:
            t_df_full = t_df_full.sort_values(by="time_tag")
            t_conn.execute("DELETE FROM dscovr_m1m")
            t_df_full.to_sql("dscovr_m1m", t_conn, if_exists="append", index=False)
            t_conn.commit()
            print(f"[OK] Sorted the entire dscovr_m1m table ({len(t_df_full)} rows).")

    print("--- SWPC IMF Backup Insertion Complete ---")

if __name__ == "__main__":
    main()
