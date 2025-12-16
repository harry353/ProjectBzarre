from space_weather_warehouse import SpaceWeatherWarehouse

M1M_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dscovr_m1m (
    time_tag TEXT PRIMARY KEY,
    bt REAL,
    bx REAL,
    by REAL,
    bz REAL
);
"""

M1M_INSERT_SQL = """
INSERT OR IGNORE INTO dscovr_m1m
(time_tag, bt, bx, by, bz)
VALUES (?, ?, ?, ?, ?);
"""

M1M_COLUMNS = ["time_tag", "bt", "bx", "by", "bz"]


def ingest_imf_discovr(df, warehouse: SpaceWeatherWarehouse):
    """
    Insert IMF rows into SQLite using the provided warehouse.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(M1M_TABLE_SQL)

    payload = df.copy()
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload[M1M_COLUMNS].drop_duplicates(subset=["time_tag"])
    payload = payload.where(payload.notna(), None)

    return warehouse.insert_rows(M1M_INSERT_SQL, payload.values.tolist())
