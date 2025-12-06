"""SQLite ingestion helpers for GOES flare archive records."""

import numpy as np

FLARE_ARCHIVE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS goes_flares_archive (
    flare_id TEXT,
    event_time TEXT,
    flare_class TEXT,
    peak_flux_wm2 REAL,
    status TEXT,
    xrsb_flux REAL,
    background_flux REAL,
    integrated_flux REAL,
    satellite TEXT,
    source_day TEXT,
    file_url TEXT,
    PRIMARY KEY (flare_id, event_time)
);
"""

FLARE_ARCHIVE_INSERT_SQL = """
INSERT OR REPLACE INTO goes_flares_archive
(flare_id, event_time, flare_class, peak_flux_wm2, status, xrsb_flux,
 background_flux, integrated_flux, satellite, source_day, file_url)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def ingest_flares_archive(df, warehouse):
    if df.empty:
        return 0

    warehouse.ensure_table(FLARE_ARCHIVE_TABLE_SQL)

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
                _coerce(row.get("satellite")),
                _coerce(row.get("source_day")),
                _coerce(row.get("file_url")),
            ]
        )

    return warehouse.insert_rows(FLARE_ARCHIVE_INSERT_SQL, payload)


def _coerce(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    return value


def _ts(value):
    if value is None:
        return None
    return getattr(value, "isoformat", lambda: str(value))()
