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
    halo_class TEXT
);
"""

CME_INSERT_SQL = """
INSERT OR REPLACE INTO lasco_cme_catalog
(event_id, catalog_month, cme_number, time_tag, dt_minutes, position_angle,
 angular_width, median_velocity, velocity_variation, min_velocity, max_velocity, halo_class)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
]


def ingest_cme_catalog(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist CME catalogue rows into SQLite via the shared warehouse helper.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(CME_TABLE_SQL)

    payload = df.copy().reindex(columns=CME_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload["catalog_month"] = payload["catalog_month"].astype(str)
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(CME_INSERT_SQL, rows)
