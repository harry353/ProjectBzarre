from space_weather_warehouse import SpaceWeatherWarehouse

DST_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dst_index (
    time_tag TEXT PRIMARY KEY,
    dst REAL
);
"""

DST_INSERT_SQL = """
INSERT OR REPLACE INTO dst_index (time_tag, dst)
VALUES (?, ?);
"""


def ingest_dst(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist Dst index rows into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(DST_TABLE_SQL)

    payload = df.copy()
    payload = payload.reindex(columns=["time_tag", "dst"])
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["dst"]
        value = float(value) if value is not None else None
        rows.append((row["time_tag"], value))

    return warehouse.insert_rows(DST_INSERT_SQL, rows)

