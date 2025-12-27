from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3

import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent
FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "sunspot_number"
    / "6_aggregate"
    / "sunspot_number_agg_eng.db"
)
FEATURES_TABLE = "features_agg"
OUTPUT_DB = STAGE_DIR / "sunspot_number_agg_eng_split.db"
TRAIN_TABLE = "sunspot_train"
VAL_TABLE = "sunspot_validation"
TEST_TABLE = "sunspot_test"

DEFAULT_WINDOWS = {
    "train": ("1999-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-11-30"),
}
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1D").replace("H", "h")


def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Feature dataframe must have a DateTimeIndex.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if AGG_FREQ == "1D":
        df.index = df.index.normalize()
    return df


def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.normalize() if AGG_FREQ == "1D" else ts


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


def create_sunspot_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(FEATURES_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {FEATURES_TABLE}",
            conn,
            parse_dates=["timestamp", "date"],
        )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    else:
        raise RuntimeError("Sunspot aggregate dataset missing timestamp/date column.")
    if df.empty:
        raise RuntimeError("Sunspot aggregate dataset is empty; run aggregate step first.")
    df = _prepare_index(df)

    windows = _get_windows()
    train = _slice(df, *windows["train"])
    val = _slice(df, *windows["validation"])
    test = _slice(df, *windows["test"])

    if train.empty or val.empty or test.empty:
        raise RuntimeError("Unable to create non-empty sunspot splits; adjust PREPROC_SPLIT_* windows.")

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    print(f"[OK] Sunspot aggregate train/val/test splits stored at {OUTPUT_DB}")
    return train, val, test


def main() -> None:
    create_sunspot_splits()


if __name__ == "__main__":
    main()
