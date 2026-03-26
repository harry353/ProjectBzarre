import sqlite3

import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

# DDL for the kp_index table — time_tag is the primary key (one record per 3-hour interval).
KP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kp_index (
    time_tag TEXT PRIMARY KEY,
    kp_index REAL,
    source_type TEXT
);
"""

# INSERT OR REPLACE so that realtime rows can overwrite earlier archive rows for the same timestamp.
KP_INSERT_SQL = """
INSERT OR REPLACE INTO kp_index (time_tag, kp_index, source_type)
VALUES (?, ?, ?);
"""

# Ordered list of columns aligned with the INSERT placeholder positions above.
KP_COLUMNS = ["time_tag", "kp_index", "source_type"]


def ingest_kp_index(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist Kp index values into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(KP_TABLE_SQL)
    # Add source_type via ALTER TABLE for databases created before the column existed.
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    # Default source_type to "archive" for rows that do not carry this column.
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    # Align column order and drop any extra columns not in the schema.
    payload = payload.reindex(columns=KP_COLUMNS)
    # Serialise timestamps to plain strings for SQLite TEXT storage.
    payload["time_tag"] = payload["time_tag"].astype(str)

    # Convert kp_index to Python float (or None) so SQLite stores REAL / NULL correctly.
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
    # Migrate older databases that were created before source_type was added to the schema.
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(kp_index)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE kp_index ADD COLUMN source_type TEXT")
            conn.commit()
