from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]

VECTOR_DB = PROJECT_ROOT / "daily_inference" / "build_vector" / "daily_inference_vector.db"
HISTORY_DB = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
OUTPUT_DB = BASE_DIR / "daily_inference_vector_merged.db"

TIME_COL_CANDIDATES = ("time_tag", "timestamp", "date")


def _get_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]


def _detect_time_col(df: pd.DataFrame) -> str | None:
    for candidate in TIME_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _load_recent_history_from_db(
    conn: sqlite3.Connection, table: str, cutoff: pd.Timestamp
) -> pd.DataFrame:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    time_col = next((c for c in TIME_COL_CANDIDATES if c in cols), None)
    if not time_col:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    time_expr = f"replace(substr({time_col}, 1, 19), 'T', ' ')"
    return pd.read_sql_query(
        f"SELECT * FROM {table} WHERE datetime({time_expr}) >= datetime(?)",
        conn,
        params=(cutoff_str,),
    )


def _load_tables_as_dfs(db_path: Path) -> dict[str, pd.DataFrame]:
    db_uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(db_uri, uri=True, timeout=30) as conn:
            conn.execute("PRAGMA busy_timeout = 30000;")
            tables = _get_tables(conn)
            return {table: pd.read_sql_query(f"SELECT * FROM {table}", conn) for table in tables}


def merge_recent_history() -> Path:
    if not VECTOR_DB.exists():
        raise FileNotFoundError(f"Missing vector db: {VECTOR_DB}")
    if not HISTORY_DB.exists():
        raise FileNotFoundError(f"Missing history db: {HISTORY_DB}")

    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=183)).tz_convert("UTC")

    print(f"[INFO] Loading vector tables from {VECTOR_DB}")
    vector_tables = _load_tables_as_dfs(VECTOR_DB)
    print(f"[INFO] Loaded {len(vector_tables)} vector tables")
    print(f"[INFO] Loading history tables from {HISTORY_DB}")
    history_tables: dict[str, pd.DataFrame] = {}
    history_uri = f"file:{HISTORY_DB}?mode=ro"
    with sqlite3.connect(history_uri, uri=True, timeout=30) as history_conn:
        history_conn.execute("PRAGMA busy_timeout = 30000;")
        history_table_names = set(_get_tables(history_conn))
        for table in vector_tables:
            if table not in history_table_names:
                continue
            history_tables[table] = _load_recent_history_from_db(
                history_conn, table, cutoff
            )
    print(f"[INFO] Loaded {len(history_tables)} history tables")

    with sqlite3.connect(OUTPUT_DB, timeout=30) as out_conn:
        out_conn.execute("PRAGMA busy_timeout = 30000;")
        for table, vector_df in vector_tables.items():
            print(f"[INFO] Merging table {table}")
            history_df = history_tables.get(table)
            if history_df is None:
                history_df = pd.DataFrame(columns=vector_df.columns)

            combined = pd.concat([history_df, vector_df], ignore_index=True)
            time_col = _detect_time_col(combined)
            if time_col:
                normalized = (
                    combined[time_col]
                    .astype(str)
                    .str.replace("T", " ", regex=False)
                    .str.replace("+00:00", "", regex=False)
                )
                combined[time_col] = pd.to_datetime(
                    normalized, errors="coerce", utc=True
                )
                combined = combined.dropna(subset=[time_col]).sort_values(time_col)
                combined = combined.drop_duplicates(subset=[time_col], keep="last")
                latest = combined[time_col].max() if not combined.empty else None
                print(
                    f"[INFO] {table}: history={len(history_df)} vector={len(vector_df)} "
                    f"combined={len(combined)} latest={latest}"
                )

            combined.to_sql(table, out_conn, if_exists="replace", index=False)

    print(f"[OK] Merged vector + recent history into {OUTPUT_DB}")
    return OUTPUT_DB


if __name__ == "__main__":
    merge_recent_history()
