from __future__ import annotations

import numpy as np
import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

LASCO_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cme_lasco_events (
    event_key TEXT PRIMARY KEY,
    event_date TEXT NOT NULL,
    event_time TEXT NOT NULL,
    datetime_utc TEXT NOT NULL,
    cpa REAL,
    width REAL,
    linear_speed REAL,
    initial_speed REAL,
    final_speed REAL,
    speed_20r REAL,
    acceleration REAL,
    mass REAL,
    kinetic_energy REAL,
    mpa REAL,
    remarks TEXT
);
"""

LASCO_INSERT_SQL = """
INSERT OR IGNORE INTO cme_lasco_events
(event_key, event_date, event_time, datetime_utc, cpa, width, linear_speed,
 initial_speed, final_speed, speed_20r, acceleration, mass, kinetic_energy,
 mpa, remarks)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

LASCO_COLUMNS = [
    "event_key",
    "event_date",
    "event_time",
    "datetime_utc",
    "cpa",
    "width",
    "linear_speed",
    "initial_speed",
    "final_speed",
    "speed_20r",
    "acceleration",
    "mass",
    "kinetic_energy",
    "mpa",
    "remarks",
]


def ingest_cme_lasco(df: pd.DataFrame, warehouse: SpaceWeatherWarehouse) -> int:
    """
    Store LASCO CME rows in SQLite using the provided warehouse helper.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(LASCO_TABLE_SQL)

    payload = df.copy()
    payload = payload.rename(
        columns={
            "date": "event_date",
            "time": "event_time",
            "Datetime": "datetime_utc",
            "CPA": "cpa",
            "Width": "width",
            "Linear_Speed": "linear_speed",
            "Initial_Speed": "initial_speed",
            "Final_Speed": "final_speed",
            "Speed_20R": "speed_20r",
            "Acceleration": "acceleration",
            "Mass": "mass",
            "Kinetic_Energy": "kinetic_energy",
            "MPA": "mpa",
            "Remarks": "remarks",
        }
    )
    payload = payload.drop_duplicates(subset=["event_key"])
    payload = payload.reindex(columns=LASCO_COLUMNS)
    payload["datetime_utc"] = payload["datetime_utc"].astype(str)

    rows = [_normalize_row(row) for row in payload.values.tolist()]
    return warehouse.insert_rows(LASCO_INSERT_SQL, rows)


def _normalize_row(row):
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
