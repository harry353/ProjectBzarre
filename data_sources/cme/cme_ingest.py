import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

# DDL for the LASCO CME catalogue table — uses event_id as the primary key.
CME_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS lasco_cme_catalog (
    event_id TEXT PRIMARY KEY,
    catalog_month TEXT NOT NULL,
    cme_number TEXT,
    time_tag TEXT NOT NULL,
    dt_minutes REAL,
    position_angle REAL,
    angular_width REAL,
    median_velocity REAL,
    velocity_variation REAL,
    min_velocity REAL,
    max_velocity REAL,
    halo_class TEXT,
    source_type TEXT
);
"""

# INSERT OR REPLACE so re-runs are idempotent and existing rows are updated.
CME_INSERT_SQL = """
INSERT OR REPLACE INTO lasco_cme_catalog
(event_id, catalog_month, cme_number, time_tag, dt_minutes, position_angle,
 angular_width, median_velocity, velocity_variation, min_velocity, max_velocity, halo_class, source_type)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

# Ordered list of columns aligned with the INSERT placeholder positions above.
CME_COLUMNS = [
    "event_id",
    "catalog_month",
    "cme_number",
    "time_tag",
    "dt_minutes",
    "position_angle",
    "angular_width",
    "median_velocity",
    "velocity_variation",
    "min_velocity",
    "max_velocity",
    "halo_class",
    "source_type",
]


def ingest_cme_catalog(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist CME catalogue rows into SQLite via the shared warehouse helper.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(CME_TABLE_SQL)
    # Add the source_type column via ALTER TABLE if it was not created by the DDL above.
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    # Default source_type to "archive" for rows that do not carry this column.
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=CME_COLUMNS)
    # Serialise timestamps and catalog_month to plain strings for SQLite TEXT storage.
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload["catalog_month"] = payload["catalog_month"].astype(str)
    # Replace pandas NA with Python None so SQLite stores NULL correctly.
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(CME_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    # Migrate older databases that were created before source_type was added to the schema.
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lasco_cme_catalog)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE lasco_cme_catalog ADD COLUMN source_type TEXT")
            conn.commit()
