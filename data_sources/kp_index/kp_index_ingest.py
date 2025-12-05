import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

KP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kp_index (
    time_tag TEXT PRIMARY KEY,
    kp_index REAL
);
"""

KP_INSERT_SQL = """
INSERT OR REPLACE INTO kp_index (time_tag, kp_index)
VALUES (?, ?);
"""

KP_COLUMNS = ["time_tag", "kp_index"]


def ingest_kp_index(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist Kp index values into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(KP_TABLE_SQL)

    payload = df.copy().reindex(columns=KP_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["kp_index"]
        if pd.isna(value):
            value = None
        else:
            value = float(value)
        rows.append((row["time_tag"], value))

    return warehouse.insert_rows(KP_INSERT_SQL, rows)

