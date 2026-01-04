import sqlite3

import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

SUNSPOT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sunspot_numbers (
    time_tag TEXT PRIMARY KEY,
    sunspot_number REAL,
    source_type TEXT
);
"""

SUNSPOT_INSERT_SQL = """
INSERT OR REPLACE INTO sunspot_numbers (time_tag, sunspot_number, source_type)
VALUES (?, ?, ?);
"""

SUNSPOT_COLUMNS = ["time_tag", "sunspot_number", "source_type"]


def ingest_sunspot_numbers(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist sunspot numbers into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(SUNSPOT_TABLE_SQL)
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=SUNSPOT_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["sunspot_number"]
        if pd.isna(value):
            value = None
        else:
            value = float(value)
        rows.append((row["time_tag"], value, row["source_type"]))

    return warehouse.insert_rows(SUNSPOT_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sunspot_numbers)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE sunspot_numbers ADD COLUMN source_type TEXT")
            conn.commit()
