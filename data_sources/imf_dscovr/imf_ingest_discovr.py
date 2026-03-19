import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

# DDL for the DSCOVR M1M table — time_tag is the primary key (one record per minute).
M1M_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dscovr_m1m (
    time_tag TEXT PRIMARY KEY,
    bt REAL,
    bx REAL,
    by REAL,
    bz REAL,
    source_type TEXT
);
"""

# INSERT OR IGNORE avoids overwriting existing rows so archive data is never replaced by a re-run.
M1M_INSERT_SQL = """
INSERT OR IGNORE INTO dscovr_m1m
(time_tag, bt, bx, by, bz, source_type)
VALUES (?, ?, ?, ?, ?, ?);
"""

# Ordered list of columns aligned with the INSERT placeholder positions above.
M1M_COLUMNS = ["time_tag", "bt", "bx", "by", "bz", "source_type"]


def ingest_imf_discovr(df, warehouse: SpaceWeatherWarehouse):
    """
    Insert IMF rows into SQLite using the provided warehouse.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(M1M_TABLE_SQL)
    # Add source_type via ALTER TABLE for databases created before the column existed.
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    # Default source_type to "archive" for rows that do not carry this column.
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    # Serialise timestamps to plain strings for SQLite TEXT storage.
    payload["time_tag"] = payload["time_tag"].astype(str)
    # Deduplicate on timestamp before inserting to avoid hitting the PRIMARY KEY constraint.
    payload = payload[M1M_COLUMNS].drop_duplicates(subset=["time_tag"])
    # Replace pandas NA with Python None so SQLite stores NULL correctly.
    payload = payload.where(payload.notna(), None)

    return warehouse.insert_rows(M1M_INSERT_SQL, payload.values.tolist())


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    # Migrate older databases that were created before source_type was added to the schema.
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(dscovr_m1m)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE dscovr_m1m ADD COLUMN source_type TEXT")
            conn.commit()
