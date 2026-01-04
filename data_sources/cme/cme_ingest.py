import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

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

CME_INSERT_SQL = """
INSERT OR REPLACE INTO lasco_cme_catalog
(event_id, catalog_month, cme_number, time_tag, dt_minutes, position_angle,
 angular_width, median_velocity, velocity_variation, min_velocity, max_velocity, halo_class, source_type)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

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
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=CME_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload["catalog_month"] = payload["catalog_month"].astype(str)
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(CME_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lasco_cme_catalog)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE lasco_cme_catalog ADD COLUMN source_type TEXT")
            conn.commit()
