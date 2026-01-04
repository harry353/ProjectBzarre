from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline.utils import write_sqlite_table  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parent
WAREHOUSE_DB = STAGE_DIR.parents[1] / "space_weather.db"
OUTPUT_DB = STAGE_DIR / "flares_comb.db"
OUTPUT_TABLE = "goes_flares_combined"
GOES_TABLES = ["goes_flares_archive", "goes_flares"]


def _load_goes_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    if "event_time" not in df.columns:
        raise RuntimeError(f"Table '{table}' missing 'event_time' column.")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"])
    df = df.set_index("event_time").sort_index()
    df["source_table"] = table
    return df


def combine_goes_flares() -> pd.DataFrame:
    if not WAREHOUSE_DB.exists():
        raise FileNotFoundError(f"Warehouse database not found at {WAREHOUSE_DB}")

    with sqlite3.connect(WAREHOUSE_DB) as conn:
        missing = [tbl for tbl in GOES_TABLES if not _table_exists(conn, tbl)]
        if missing:
            raise RuntimeError(f"Missing GOES flare tables: {', '.join(missing)}")
        frames = [_load_goes_table(conn, table) for table in GOES_TABLES]

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    write_sqlite_table(combined, OUTPUT_DB, OUTPUT_TABLE)
    print(
        f"[OK] Combined GOES flare tables ({', '.join(GOES_TABLES)}) "
        f"into {OUTPUT_DB} ({len(combined)} rows)"
    )
    return combined


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cursor.fetchone() is not None


def main() -> None:
    combine_goes_flares()


if __name__ == "__main__":
    main()
