import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

RADIO_FLUX_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS radio_flux (
    time_tag TEXT PRIMARY KEY,
    observed_flux REAL,
    adjusted_flux REAL,
    ursi_flux REAL,
    source_type TEXT
);
"""

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

    warehouse.ensure_table(RADIO_FLUX_TABLE_SQL)
    _ensure_source_type_column(warehouse)
    payload = df.copy()
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=["time_tag", "observed_flux", "adjusted_flux", "ursi_flux", "source_type"])
    payload["time_tag"] = payload["time_tag"].astype(str)

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
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(radio_flux)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE radio_flux ADD COLUMN source_type TEXT")
            conn.commit()


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
