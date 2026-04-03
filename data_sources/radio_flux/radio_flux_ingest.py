import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

# time_tag is the primary key so duplicate timestamps are replaced on re-ingest
RADIO_FLUX_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS radio_flux (
    time_tag TEXT PRIMARY KEY,
    observed_flux REAL,
    adjusted_flux REAL,
    ursi_flux REAL,
    source_type TEXT
);
"""

# UPSERT statement — existing rows for a given timestamp are overwritten
RADIO_FLUX_INSERT_SQL = """
INSERT OR REPLACE INTO radio_flux (time_tag, observed_flux, adjusted_flux, ursi_flux, source_type)
VALUES (?, ?, ?, ?, ?);
"""


def ingest_radio_flux(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist radio flux readings into SQLite.
    """
    if df.empty:
        return 0

    # Create the table if this is the first ingest run
    warehouse.ensure_table(RADIO_FLUX_TABLE_SQL)
    # Migrate databases that were created before source_type was added to the schema
    _ensure_source_type_column(warehouse)
    payload = df.copy()
    # Default source_type to "archive" when the column is absent from the incoming dataframe
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    # Also fill any NaN values in an existing source_type column
    payload["source_type"] = payload["source_type"].fillna("archive")
    # Guarantee column order matches the INSERT statement's positional parameters
    payload = payload.reindex(columns=["time_tag", "observed_flux", "adjusted_flux", "ursi_flux", "source_type"])
    # SQLite stores datetimes as text; convert Timestamps to ISO-format strings
    payload["time_tag"] = payload["time_tag"].astype(str)

    # Build the list of tuples expected by the warehouse's batch insert helper
    rows = []
    for _, row in payload.iterrows():
        rows.append(
            (
                row["time_tag"],
                _to_float(row["observed_flux"]),
                _to_float(row["adjusted_flux"]),
                _to_float(row["ursi_flux"]),
                row["source_type"],
            )
        )

    return warehouse.insert_rows(RADIO_FLUX_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    # PRAGMA table_info returns one row per column; extract the name field (index 1)
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(radio_flux)")}
        # Only alter the table if the column is genuinely missing (idempotent migration)
        if "source_type" not in cols:
            conn.execute("ALTER TABLE radio_flux ADD COLUMN source_type TEXT")
            conn.commit()


def _to_float(value):
    # Propagate SQL NULLs directly rather than storing "None" as a string
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        # Non-numeric sentinel values (e.g. missing-data flags) are stored as NULL
        return None
