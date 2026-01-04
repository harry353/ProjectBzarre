import sqlite3

import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

KP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kp_index (
    time_tag TEXT PRIMARY KEY,
    kp_index REAL,
    source_type TEXT
);
"""

KP_INSERT_SQL = """
INSERT OR REPLACE INTO kp_index (time_tag, kp_index, source_type)
VALUES (?, ?, ?);
"""

KP_COLUMNS = ["time_tag", "kp_index", "source_type"]


def ingest_kp_index(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist Kp index values into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(KP_TABLE_SQL)
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=KP_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["kp_index"]
        if pd.isna(value):
            value = None
        else:
            value = float(value)
        rows.append((row["time_tag"], value, row["source_type"]))

    return warehouse.insert_rows(KP_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(kp_index)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE kp_index ADD COLUMN source_type TEXT")
            conn.commit()
