import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

IMF_ACE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ace_mfi (
    time_tag TEXT PRIMARY KEY,
    bx_gsm REAL,
    by_gsm REAL,
    bz_gsm REAL,
    bt REAL,
    source_type TEXT
);
"""

IMF_ACE_INSERT_SQL = """
INSERT OR REPLACE INTO ace_mfi (time_tag, bx_gsm, by_gsm, bz_gsm, bt, source_type)
VALUES (?, ?, ?, ?, ?, ?);
"""

IMF_ACE_COLUMNS = ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "bt", "source_type"]


def ingest_imf_ace(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist ACE IMF rows into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(IMF_ACE_TABLE_SQL)
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=IMF_ACE_COLUMNS)
    payload["time_tag"] = payload["time_tag"].astype(str)
    payload = payload.where(payload.notna(), None)

    rows = payload.to_records(index=False).tolist()
    return warehouse.insert_rows(IMF_ACE_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(ace_mfi)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE ace_mfi ADD COLUMN source_type TEXT")
            conn.commit()
