from __future__ import annotations

from space_weather_warehouse import SpaceWeatherWarehouse

DSCOVR_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dscovr_f1m (
    time_tag TEXT PRIMARY KEY,
    density REAL,
    speed REAL,
    temperature REAL
);
"""

DSCOVR_INSERT_SQL = """
INSERT OR IGNORE INTO dscovr_f1m
(time_tag, density, speed, temperature)
VALUES (?, ?, ?, ?);
"""

COLUMNS = ["time_tag", "density", "speed", "temperature"]


def ingest_solar_wind_dscovr(df, warehouse: SpaceWeatherWarehouse) -> int:
    if df.empty:
        return 0

    warehouse.ensure_table(DSCOVR_TABLE_SQL)
    payload = (
        df.copy()
        .reindex(columns=COLUMNS)
        .drop_duplicates(subset=["time_tag"])
    )
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload.where(payload.notna(), None)
    return warehouse.insert_rows(DSCOVR_INSERT_SQL, payload.values.tolist())
