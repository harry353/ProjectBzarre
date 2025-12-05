from space_weather_warehouse import SpaceWeatherWarehouse

SW_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dscovr_f1m (
    time_tag TEXT PRIMARY KEY,
    density REAL,
    speed REAL,
    temperature REAL
);
"""

SW_INSERT_SQL = """
INSERT OR IGNORE INTO dscovr_f1m
(time_tag, density, speed, temperature)
VALUES (?, ?, ?, ?);
"""

SW_COLUMNS = ["time_tag", "density", "speed", "temperature"]


def ingest_solar_wind(df, warehouse: SpaceWeatherWarehouse):
    """
    Insert DSCOVR F1M rows into SQLite via the provided warehouse.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(SW_TABLE_SQL)

    payload = df.copy()
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload[SW_COLUMNS].drop_duplicates(subset=["time_tag"])
    payload = payload.where(payload.notna(), None)

    return warehouse.insert_rows(SW_INSERT_SQL, payload.values.tolist())
