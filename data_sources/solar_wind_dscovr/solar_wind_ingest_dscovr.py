from __future__ import annotations

import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

DSCOVR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dscovr_f1m (
    time_tag TEXT PRIMARY KEY,
    density REAL,
    speed REAL,
    temperature REAL,
    source_type TEXT
);
"""

DSCOVR_INSERT_SQL = """
INSERT OR IGNORE INTO dscovr_f1m
(time_tag, density, speed, temperature, source_type)
VALUES (?, ?, ?, ?, ?);
"""

COLUMNS = ["time_tag", "density", "speed", "temperature", "source_type"]


def ingest_solar_wind_dscovr(df, warehouse: SpaceWeatherWarehouse) -> int:
    if df.empty:
        return 0

    warehouse.ensure_table(DSCOVR_TABLE_SQL)
    _ensure_source_type_column(warehouse)
    payload = (
        df.copy()
        .assign(source_type=df.get("source_type", "archive"))
        .reindex(columns=COLUMNS)
        .drop_duplicates(subset=["time_tag"])
    )
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload.where(payload.notna(), None)
    return warehouse.insert_rows(DSCOVR_INSERT_SQL, payload.values.tolist())


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(dscovr_f1m)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE dscovr_f1m ADD COLUMN source_type TEXT")
            conn.commit()
