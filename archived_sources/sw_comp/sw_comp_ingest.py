from space_weather_warehouse import SpaceWeatherWarehouse

SW_COMP_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ace_swics_composition (
    time_tag TEXT PRIMARY KEY,
    o7_o6 REAL,
    c6_c5 REAL,
    avg_fe_charge REAL,
    fe_to_o REAL
);
"""

SW_COMP_INSERT_SQL = """
INSERT OR REPLACE INTO ace_swics_composition
(time_tag, o7_o6, c6_c5, avg_fe_charge, fe_to_o)
VALUES (?, ?, ?, ?, ?);
"""

SW_COMP_COLUMNS = [
    "time_tag",
    "o7_o6",
    "c6_c5",
    "avg_fe_charge",
    "fe_to_o",
]


def ingest_sw_comp(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist ACE/SWICS composition ratios into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(SW_COMP_TABLE_SQL)

    payload = df.copy().reindex(columns=SW_COMP_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(SW_COMP_INSERT_SQL, rows)
