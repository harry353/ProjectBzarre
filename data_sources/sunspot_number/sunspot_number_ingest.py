import pandas as pd

from space_weather_warehouse import SpaceWeatherWarehouse

SUNSPOT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sunspot_numbers (
    time_tag TEXT PRIMARY KEY,
    sunspot_number REAL
);
"""

SUNSPOT_INSERT_SQL = """
INSERT OR REPLACE INTO sunspot_numbers (time_tag, sunspot_number)
VALUES (?, ?);
"""

SUNSPOT_COLUMNS = ["time_tag", "sunspot_number"]


def ingest_sunspot_numbers(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist sunspot numbers into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(SUNSPOT_TABLE_SQL)

    payload = df.copy()
    payload = payload.reindex(columns=SUNSPOT_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["sunspot_number"]
        if pd.isna(value):
            value = None
        else:
            value = float(value)
        rows.append((row["time_tag"], value))

    return warehouse.insert_rows(SUNSPOT_INSERT_SQL, rows)

