from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import sqlite3

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_DIR = Path(__file__).resolve().parent
CATALOG_DB = STAGE_DIR.parents[1] / "space_weather.db"
CATALOG_TABLE = "lasco_cme_catalog"
OUTPUT_DB = STAGE_DIR / "cme_catalog_split.db"

TRAIN_TABLE = "cme_catalog_train"
VAL_TABLE = "cme_catalog_validation"
TEST_TABLE = "cme_catalog_test"

DEFAULT_WINDOWS = {
    "train": ("1999-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-11-30"),
}


def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("CME catalog must have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _get_windows() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    env = os.environ
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, (start_default, end_default) in DEFAULT_WINDOWS.items():
        start = env.get(f"PREPROC_SPLIT_{split.upper()}_START", start_default)
        end = env.get(f"PREPROC_SPLIT_{split.upper()}_END", end_default)
        windows[split] = (_parse_date(start), _parse_date(end))
    return windows


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df.index.tz is not None:
        if start.tzinfo is None:
            start = start.tz_localize(df.index.tz)
        if end.tzinfo is None:
            end = end.tz_localize(df.index.tz)
    elif start.tzinfo is not None or end.tzinfo is not None:
        start = start.tz_convert(None) if start.tzinfo is not None else start
        end = end.tz_convert(None) if end.tzinfo is not None else end
    return df.loc[(df.index >= start) & (df.index <= end)]


def _load_catalog() -> pd.DataFrame:
    with sqlite3.connect(CATALOG_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {CATALOG_TABLE}",
            conn,
            parse_dates=["time_tag"],
        )
    if df.empty:
        return df
    if "time_tag" in df.columns:
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
        df = df.dropna(subset=["time_tag"])
        return df.set_index("time_tag").sort_index()
    raise RuntimeError("CME catalog missing time_tag column.")


def create_cme_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _load_catalog()
    if df.empty:
        raise RuntimeError("CME catalog dataset is empty.")
    df = _prepare_index(df)

    windows = _get_windows()
    train = _slice(df, *windows["train"])
    val = _slice(df, *windows["validation"])
    test = _slice(df, *windows["test"])

    if train.empty or val.empty or test.empty:
        raise RuntimeError("Unable to create non-empty temporal splits; adjust PREPROC_SPLIT_* windows.")

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    print(f"[OK] CME catalog splits stored at {OUTPUT_DB}")
    return train, val, test


def main() -> None:
    create_cme_splits()


if __name__ == "__main__":
    main()
