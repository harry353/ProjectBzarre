from space_weather_warehouse import SpaceWeatherWarehouse

IMF_ACE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ace_mfi (
    time_tag TEXT PRIMARY KEY,
    bx_gse REAL,
    by_gse REAL,
    bz_gse REAL,
    bt REAL
);
"""

IMF_ACE_INSERT_SQL = """
INSERT OR REPLACE INTO ace_mfi (time_tag, bx_gse, by_gse, bz_gse, bt)
VALUES (?, ?, ?, ?, ?);
"""

IMF_ACE_COLUMNS = ["time_tag", "bx_gse", "by_gse", "bz_gse", "bt"]


def ingest_imf_ace(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist ACE IMF rows into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(IMF_ACE_TABLE_SQL)

    payload = df.copy().reindex(columns=IMF_ACE_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(IMF_ACE_INSERT_SQL, rows)
