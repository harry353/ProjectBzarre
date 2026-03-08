from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
OUTPUT_DB = PROJECT_ROOT / "inference" / "space_weather_last_6m.db"
HOURS_BACK = 24 * 30 * 6

TIME_COLUMNS = {
    "ace_mfi": "time_tag",
    "ace_swepam": "time_tag",
    "dscovr_f1m": "time_tag",
    "dscovr_m1m": "time_tag",
    "dst_index": "time_tag",
    "kp_index": "time_tag",
    "lasco_cme_catalog": "time_tag",
    "radio_flux": "time_tag",
    "sunspot_numbers": "time_tag",
}
ROW_BATCH_SIZE = 5000


# Parse datetime strings from various formats (ISO, UTC, SQL) into datetime objects
def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    # Normalize 'Z' suffix for Python ISO parser
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # Fallback to common database string formats if ISO fails
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


# Scan all monitored tables in source DB to find the most recent global timestamp
def _latest_timestamp(conn: sqlite3.Connection) -> datetime:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    latest = None
    for table in tables:
        col = TIME_COLUMNS.get(table)
        if not col:
            continue
        # Retrieve the maximum timestamp value for the current table
        row = conn.execute(
            f"SELECT MAX(datetime({col})) FROM {table} WHERE {col} IS NOT NULL"
        ).fetchone()
        ts = _parse_dt(row[0] if row else None)
        if ts and (latest is None or ts > latest):
            latest = ts
    if latest is None:
        raise RuntimeError("No timestamps found in source database.")
    return latest


# Copy records for a specific table from source to destination within a time window
def _copy_table(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
    cutoff: str | None,
    end_cutoff: str | None = None,
) -> int:
    # Replicate the original table schema in the destination database
    ddl_row = src.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not ddl_row or not ddl_row[0]:
        return 0

    dst.execute(ddl_row[0])

    cols = [row[1] for row in src.execute(f"PRAGMA table_info({table})")]
    placeholders = ",".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"

    time_col = TIME_COLUMNS.get(table)
    # Construct filtered SELECT statement if time tracking metadata is present
    if time_col and cutoff:
        select_sql = (
            f"SELECT * FROM {table} WHERE {time_col} IS NOT NULL "
            f"AND datetime({time_col}) >= datetime(?)"
        )
        params = [cutoff]
        if end_cutoff:
            select_sql += f" AND datetime({time_col}) <= datetime(?)"
            params.append(end_cutoff)
    else:
        select_sql = f"SELECT * FROM {table}"
        params = []

    count = 0
    batch = []
    # Stream rows from source and perform bulk insertions to destination
    for row in src.execute(select_sql, params):
        batch.append(row)
        if len(batch) >= ROW_BATCH_SIZE:
            dst.executemany(insert_sql, batch)
            count += len(batch)
            batch.clear()
    if batch:
        dst.executemany(insert_sql, batch)
        count += len(batch)
    return count


# Main logic to create a sliding-window subset of the space weather database
def main(output_db: Path | None = None, overwrite: bool = False) -> Path:
    output_path = output_db or OUTPUT_DB
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(SOURCE_DB) as src:
        # Calculate the 6-month cutoff based on the latest available data
        latest = _latest_timestamp(src)
        cutoff_dt = latest - timedelta(hours=HOURS_BACK)
        cutoff = cutoff_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Select tables that have defined time columns
        tables = [
            row[0]
            for row in src.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if row[0] in TIME_COLUMNS
        ]

        with sqlite3.connect(output_path) as dst:
            total_rows = 0
            for table in tables:
                # Process each table sequentially
                count = _copy_table(src, dst, table, cutoff)
                total_rows += count
                print(f"[OK] {table}: {count:,} rows")
            dst.commit()

    print(f"[OK] Latest cutoff: {cutoff}")
    print(f"[OK] Output DB: {output_path} ({total_rows:,} rows total)")
    return output_path


if __name__ == "__main__":
    main()
