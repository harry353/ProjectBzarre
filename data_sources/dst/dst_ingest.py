import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

# DDL for the Dst index table — time_tag is the primary key (one reading per hour).
DST_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dst_index (
    time_tag TEXT PRIMARY KEY,
    dst REAL,
    source_type TEXT
);
"""

# INSERT OR REPLACE keeps the table idempotent across repeated ingestion runs.
DST_INSERT_SQL = """
INSERT OR REPLACE INTO dst_index (time_tag, dst, source_type)
VALUES (?, ?, ?);
"""


def ingest_dst(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist Dst index rows into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(DST_TABLE_SQL)
    # Add source_type via ALTER TABLE for databases created before the column existed.
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    # Default source_type to "archive" for rows that do not carry this column.
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=["time_tag", "dst", "source_type"])
    # Serialise timestamps to plain strings for SQLite TEXT storage.
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["dst"]
        # Explicitly cast to float so SQLite stores REAL; keep None as SQL NULL.
        value = float(value) if value is not None else None
        rows.append((row["time_tag"], value, row["source_type"]))

    return warehouse.insert_rows(DST_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    # Migrate older databases that were created before source_type was added to the schema.
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(dst_index)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE dst_index ADD COLUMN source_type TEXT")
            conn.commit()
