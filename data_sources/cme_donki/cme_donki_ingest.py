import numpy as np

from space_weather_warehouse import SpaceWeatherWarehouse

CME_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cme_analysis (
    associatedCMEID TEXT PRIMARY KEY,
    time21_5 TEXT,
    latitude REAL,
    longitude REAL,
    halfAngle REAL,
    speed REAL,
    type TEXT,
    catalog TEXT,
    note TEXT
);
"""

CME_INSERT_SQL = """
INSERT OR IGNORE INTO cme_analysis
(associatedCMEID, time21_5, latitude, longitude, halfAngle,
 speed, type, catalog, note)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

CME_COLUMNS = [
    "associatedCMEID",
    "time21_5",
    "latitude",
    "longitude",
    "halfAngle",
    "speed",
    "type",
    "catalog",
    "note",
]


def ingest_cme(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist CME rows into SQLite via the provided warehouse.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(CME_TABLE_SQL)

    payload = df.copy()
    payload = payload.drop_duplicates(subset=["associatedCMEID"])
    payload = payload.reindex(columns=CME_COLUMNS)
    payload["time21_5"] = payload["time21_5"].astype(str)

    rows = [_to_py(row) for row in payload.values.tolist()]
    return warehouse.insert_rows(CME_INSERT_SQL, rows)


def _to_py(row):
    converted = []
    for value in row:
        if value is None:
            converted.append(None)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            converted.append(float(value))
        elif isinstance(value, (np.integer, np.int32, np.int64)):
            converted.append(int(value))
        else:
            converted.append(value)
    return converted
