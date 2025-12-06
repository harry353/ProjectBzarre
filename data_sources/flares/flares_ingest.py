"""SQLite ingestion for GOES flare summary rows."""

import numpy as np

FLARE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS goes_flares (
    flare_id TEXT PRIMARY KEY,
    event_time TEXT,
    flare_class TEXT,
    peak_flux_wm2 REAL,
    status TEXT,
    xrsb_flux REAL,
    background_flux REAL,
    integrated_flux REAL,
    source_day TEXT,
    satellite TEXT
);
"""

FLARE_INSERT_SQL = """
INSERT OR REPLACE INTO goes_flares
(flare_id, event_time, flare_class, peak_flux_wm2, status, xrsb_flux,
 background_flux, integrated_flux, source_day, satellite)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def ingest_flares(df, warehouse):
    """Insert DataFrame rows into SQLite via the provided warehouse."""

    if df.empty:
        return 0

    warehouse.ensure_table(FLARE_TABLE_SQL)

    payload = []
    for _, row in df.iterrows():
        payload.append(
            [
                _coerce(row.get("flare_id")),
                _coerce(_ts(row.get("event_time"))),
                _coerce(row.get("flare_class")),
                _coerce(row.get("peak_flux_wm2")),
                _coerce(row.get("status")),
                _coerce(row.get("xrsb_flux")),
                _coerce(row.get("background_flux")),
                _coerce(row.get("integrated_flux")),
                _coerce(row.get("source_day")),
                _coerce(row.get("satellite")),
            ]
        )

    return warehouse.insert_rows(FLARE_INSERT_SQL, payload)


def _coerce(value):
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _ts(value):
    if value is None:
        return None
    return getattr(value, "isoformat", lambda: str(value))()
