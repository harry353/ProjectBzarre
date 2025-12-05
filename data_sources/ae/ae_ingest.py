from space_weather_warehouse import SpaceWeatherWarehouse

AE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ae_indices (
    time_tag TEXT PRIMARY KEY,
    al REAL,
    au REAL,
    ae REAL,
    ao REAL
);
"""

AE_INSERT_SQL = """
INSERT OR REPLACE INTO ae_indices (time_tag, al, au, ae, ao)
VALUES (?, ?, ?, ?, ?);
"""

AE_COLUMNS = ["time_tag", "al", "au", "ae", "ao"]


def ingest_ae(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist AE/AL/AU/AO readings into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(AE_TABLE_SQL)
    payload = df.copy().reindex(columns=AE_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        rows.append(
            (
                row["time_tag"],
                _to_float(row["al"]),
                _to_float(row["au"]),
                _to_float(row["ae"]),
                _to_float(row["ao"]),
            )
        )

    return warehouse.insert_rows(AE_INSERT_SQL, rows)


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

