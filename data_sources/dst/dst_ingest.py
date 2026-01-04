import sqlite3

from space_weather_warehouse import SpaceWeatherWarehouse

DST_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dst_index (
    time_tag TEXT PRIMARY KEY,
    dst REAL,
    source_type TEXT
);
"""

DST_INSERT_SQL = """
INSERT OR REPLACE INTO dst_index (time_tag, dst, source_type)
VALUES (?, ?, ?);
"""


def ingest_dst(df, warehouse: SpaceWeatherWarehouse):
    """
    Persist Dst index rows into SQLite.
    """
    if df.empty:
        return 0

    warehouse.ensure_table(DST_TABLE_SQL)
    _ensure_source_type_column(warehouse)

    payload = df.copy()
    if "source_type" not in payload.columns:
        payload["source_type"] = "archive"
    payload["source_type"] = payload["source_type"].fillna("archive")
    payload = payload.reindex(columns=["time_tag", "dst", "source_type"])
    payload["time_tag"] = payload["time_tag"].astype(str)

    rows = []
    for _, row in payload.iterrows():
        value = row["dst"]
        value = float(value) if value is not None else None
        rows.append((row["time_tag"], value, row["source_type"]))

    return warehouse.insert_rows(DST_INSERT_SQL, rows)


def _ensure_source_type_column(warehouse: SpaceWeatherWarehouse) -> None:
    with sqlite3.connect(warehouse.db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(dst_index)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE dst_index ADD COLUMN source_type TEXT")
            conn.commit()
