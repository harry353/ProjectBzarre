import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

# DDL for the ACE MFI table — time_tag is the primary key (one record per measurement epoch).
IMF_ACE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ace_mfi (
    time_tag TEXT PRIMARY KEY,
    bx_gsm REAL,
    by_gsm REAL,
    bz_gsm REAL,
    bt REAL,
    source_type TEXT
);
"""

# INSERT OR REPLACE keeps the table idempotent across repeated ingestion runs.
IMF_ACE_INSERT_SQL = """
INSERT OR REPLACE INTO ace_mfi (time_tag, bx_gsm, by_gsm, bz_gsm, bt, source_type)
VALUES (?, ?, ?, ?, ?, ?);
"""

# Ordered list of columns aligned with the INSERT placeholder positions above.
IMF_ACE_COLUMNS = ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt", "source_type"]


def ingest_imf_ace(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist ACE IMF rows into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(IMF_ACE_TABLE_SQL)
    # Add source_type via ALTER TABLE for databases created before the column existed.
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    # Default source_type to "archive" for rows that do not carry this column.
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=IMF_ACE_COLUMNS)
    # Serialise timestamps to plain strings for SQLite TEXT storage.
    payload["time_tag"] = payload["time_tag"].astype(str)
    # Replace pandas NA with Python None so SQLite stores NULL correctly.
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(IMF_ACE_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    # Migrate older databases that were created before source_type was added to the schema.
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ace_mfi)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE ace_mfi ADD COLUMN source_type TEXT")
            conn.commit()
