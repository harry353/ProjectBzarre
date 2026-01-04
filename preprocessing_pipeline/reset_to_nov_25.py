from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
STATUS_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "data_source_status.csv"

CUTOFF = datetime(2025, 12, 1, tzinfo=timezone.utc)
TIME_COL_CANDIDATES = ("time_tag", "timestamp", "date")
CLASS_TABLE_MAP = {
    "DstDataSource": "dst_index",
    "CMEDataSource": "lasco_cme_catalog",
    "KpIndexDataSource": "kp_index",
    "SunspotNumberDataSource": "sunspot_numbers",
    "RadioFluxDataSource": "radio_flux",
    "XRayFluxGOESDataSource": "xray_flux",
    "SolarWindDSCOVRDataSource": "dscovr_f1m",
    "IMFDSCOVRDataSource": "dscovr_m1m",
    "SolarWindACEDataSource": "ace_swepam",
    "IMFACEDataSource": "ace_mfi",
}


def _get_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]


def _detect_time_col(conn: sqlite3.Connection, table: str) -> str | None:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    for candidate in TIME_COL_CANDIDATES:
        if candidate in cols:
            return candidate
    return None


def _delete_from_table(conn: sqlite3.Connection, table: str, time_col: str) -> int:
    cutoff_str = CUTOFF.strftime("%Y-%m-%d %H:%M:%S")
    time_expr = f"replace(substr({time_col}, 1, 19), 'T', ' ')"
    cur = conn.execute(
        f"DELETE FROM {table} WHERE datetime({time_expr}) >= datetime(?)",
        (cutoff_str,),
    )
    return cur.rowcount


def reset_to_nov_25() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA busy_timeout = 30000;")
        tables = _get_tables(conn)
        total_deleted = 0
        for table in tables:
            time_col = _detect_time_col(conn, table)
            if not time_col:
                continue
            deleted = _delete_from_table(conn, table, time_col)
            total_deleted += deleted
            print(f"[OK] {table}: deleted {deleted} rows on/after 2025-12-01")
        print(f"[OK] Total rows deleted: {total_deleted}")

        _reset_status_file(conn)


def _reset_status_file(conn: sqlite3.Connection) -> None:
    from database_builder.discovery import load_data_source_classes

    classes = load_data_source_classes()
    class_names = [cls.__name__ for cls in classes]
    tracker: dict[str, datetime | None] = {name: None for name in class_names}

    for class_name in class_names:
        table = CLASS_TABLE_MAP.get(class_name)
        if not table:
            continue
        time_col = _detect_time_col(conn, table)
        if not time_col:
            continue
        time_expr = f"replace(substr({time_col}, 1, 19), 'T', ' ')"
        row = conn.execute(
            f"SELECT MAX(datetime({time_expr})) FROM {table}"
        ).fetchone()
        if row and row[0]:
            tracker[class_name] = datetime.fromisoformat(row[0])

    with STATUS_PATH.open("w", encoding="utf-8", newline="") as fh:
        fh.write("source,latest_timestamp\n")
        for name in class_names:
            value = tracker.get(name)
            stamp_value = value.isoformat(timespec="seconds") if value else ""
            fh.write(f"{name},{stamp_value}\n")

    print(f"[OK] Updated status file at {STATUS_PATH}")


if __name__ == "__main__":
    reset_to_nov_25()
