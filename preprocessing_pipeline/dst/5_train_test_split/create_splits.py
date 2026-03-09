from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Handle absolute path resolution with fallbacks for local and installed modes
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuration for source and destination databases
STAGE_DIR = Path(__file__).resolve().parent
FEATURES_DB = (
    STAGE_DIR.parents[1] / "dst" / "4_imputation" / "dst_aver_filt_imp.db"
)
FEATURES_TABLE = "imputed_data"
OUTPUT_DB = STAGE_DIR / "dst_imputed_split.db"

TRAIN_TABLE = "dst_train"
VAL_TABLE = "dst_validation"
TEST_TABLE = "dst_test"

# Default UTC windows for data partitioning
DEFAULT_WINDOWS: Dict[str, Tuple[str, str]] = {
    "train": ("1999-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-11-30"),
}
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1h").replace("H", "h")
SKIP_SPLITS = os.environ.get("PREPROC_SKIP_SPLITS", "").lower() in {"1", "true", "yes"}


# Standardizes various possible date column names into a DatetimeIndex
def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif "date" in df.columns:
        df = df.set_index("date")
    elif "time_tag" in df.columns:
        df = df.set_index("time_tag")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DST imputed dataset must have a DatetimeIndex.")
    # Ensure chronological order for efficient slicing
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # Normalize to midnight if operating on daily frequency
    if AGG_FREQ == "1D":
        df.index = df.index.normalize()
    return df


# Converts a date string into a Timestamp, respecting aggregation granularity
def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.normalize() if AGG_FREQ == "1D" else ts


# Retrieves partition boundaries from environment variables or defaults
def _get_windows() -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    env = os.environ
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, (start_default, end_default) in DEFAULT_WINDOWS.items():
        # Allow custom segmentation for cross-validation or specific study periods
        start = env.get(f"PREPROC_SPLIT_{split.upper()}_START", start_default)
        end = env.get(f"PREPROC_SPLIT_{split.upper()}_END", end_default)
        windows[split] = (_parse_date(start), _parse_date(end))
    return windows


# Extracts a temporal slice, handling potential UTC timezone ambiguities
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


# Partition the imputed DST dataset into training, validation, and testing tables
def create_dst_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(FEATURES_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {FEATURES_TABLE}",
            conn,
            parse_dates=["timestamp", "date", "time_tag"],
        )
    if df.empty:
        raise RuntimeError("DST imputed dataset is empty; run imputation step first.")
    df = _prepare_index(df)

    # Option to disable splitting for global normalization or debugging
    if SKIP_SPLITS:
        train = val = test = df
    else:
        windows = _get_windows()
        train = _slice(df, *windows["train"])
        val = _slice(df, *windows["validation"])
        test = _slice(df, *windows["test"])

        if train.empty or val.empty or test.empty:
            raise RuntimeError("One or more DST splits are empty; adjust PREPROC_SPLIT_* env settings or confirm coverage.")

    # Save the resulting partitions to the local staging split database
    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    print(f"[OK] DST imputed train/val/test splits stored at {OUTPUT_DB}")
    return train, val, test


def main() -> None:
    create_dst_splits()


if __name__ == "__main__":
    main()
