from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from database_builder.constants import DB_PATH

DEFAULT_DB_PATH = Path(DB_PATH)


def read_timeseries_table(
    table: str,
    time_col: str,
    value_cols: Sequence[str],
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load a timeseries table from the warehouse database."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    with sqlite3.connect(path) as conn:
        columns = ", ".join([time_col, *value_cols])
        df = pd.read_sql_query(
            f"SELECT {columns} FROM {table}",
            conn,
            parse_dates=[time_col],
        )
    df = df.dropna(subset=[time_col])
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()
    return df[value_cols]


def resample_to_hourly(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """Resample a dataframe to hourly cadence using the specified strategy."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex to resample.")
    df = df.sort_index()
    if method == "ffill":
        return df.resample("1h").ffill()
    if method == "mean":
        return df.resample("1h").mean()
    raise ValueError(f"Unsupported resample method '{method}'.")


def write_sqlite_table(
    df: pd.DataFrame,
    db_path: Path | str,
    table_name: str,
    *,
    index_label: str | None = None,
) -> None:
    """Persist a dataframe to a sqlite table, creating parent directories as needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_write = df.copy()
    idx = to_write.index
    if isinstance(idx, pd.DatetimeIndex):
        label = index_label or idx.name or "timestamp"
        to_write = to_write.reset_index().rename(columns={idx.name or "index": label})
    else:
        to_write = to_write.reset_index(drop=True)

    with sqlite3.connect(path) as conn:
        to_write.to_sql(table_name, conn, if_exists="replace", index=False)


def load_hourly_output(db_path: Path | str, table_name: str) -> pd.DataFrame:
    """Load a stage output table from disk and restore the timestamp index."""
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()

    with sqlite3.connect(path) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()

    for candidate in ("time_tag", "timestamp"):
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate], utc=True, errors="coerce")
            df = df.dropna(subset=[candidate])
            df = df.set_index(candidate).sort_index()
            return df
    return df


__all__ = [
    "DEFAULT_DB_PATH",
    "load_hourly_output",
    "read_timeseries_table",
    "resample_to_hourly",
    "write_sqlite_table",
]
