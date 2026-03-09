from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Resolve absolute path of the current script and search for project root (space_weather_api.py)
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Inject project root into system path to allow local module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_DIR = Path(__file__).resolve().parent
# Source fully imputed IMF + solar wind data
FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "5_imputation"
    / "imf_solar_wind_aver_comb_filt_imp.db"
)
FEATURES_TABLE = "imputed_data"
# Staging database for partitioned split tables
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_imputed_split.db"

TRAIN_TABLE = "imf_solar_wind_train"
VAL_TABLE = "imf_solar_wind_validation"
TEST_TABLE = "imf_solar_wind_test"

# Default temporal boundaries for standard ML evaluation
DEFAULT_WINDOWS: Dict[str, Tuple[str, str]] = {
    "train": ("1999-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-11-30"),
}
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1h").replace("H", "h")
# Option to skip partitioning (e.g., for full-period analysis)
SKIP_SPLITS = os.environ.get("PREPROC_SKIP_SPLITS", "").lower() in {"1", "true", "yes"}


# Standardizes various possible date column names into a unified DatetimeIndex
def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif "date" in df.columns:
        df = df.set_index("date")
    elif "time_tag" in df.columns:
        df = df.set_index("time_tag")
    # Cast to datetime if not already correctly typed
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        raise ValueError("Aggregated IMF + solar wind dataset has non-datetime index values.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("IMF + solar wind imputed dataset must have a DatetimeIndex.")
    # Ensure chronological order for efficient slicing
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    # Normalize to naive UTC (removing timezone info if present)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    # Align to midnight if operating on daily granularity
    if AGG_FREQ == "1D":
        df.index = df.index.normalize()
    return df


# Converts a string date into a Timestamp object, respecting frequency normalization
def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.normalize() if AGG_FREQ == "1D" else ts


# Resolves partition boundaries from environment variables with fallback to defaults
def _get_windows() -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    env = os.environ
    windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, (start_default, end_default) in DEFAULT_WINDOWS.items():
        # Allow dynamic override for cross-validation experiments
        start = env.get(f"PREPROC_SPLIT_{split.upper()}_START", start_default)
        end = env.get(f"PREPROC_SPLIT_{split.upper()}_END", end_default)
        windows[split] = (_parse_date(start), _parse_date(end))
    return windows


# Extracts a temporal slice while ensuring timezone consistency between index and query
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


# Partition the imputed dataset into standard train, validation, and test subsets
def create_imf_solar_wind_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(FEATURES_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {FEATURES_TABLE}",
            conn,
            parse_dates=["timestamp", "date", "time_tag"],
        )
    if df.empty:
        raise RuntimeError("IMF + solar wind imputed dataset is empty; run imputation step first.")
    df = _prepare_index(df)

    if SKIP_SPLITS:
        train = val = test = df
    else:
        # Segment the data using resolved temporal windows
        windows = _get_windows()
        train = _slice(df, *windows["train"])
        val = _slice(df, *windows["validation"])
        test = _slice(df, *windows["test"])

        if train.empty or val.empty or test.empty:
            raise RuntimeError("Unable to create non-empty IMF + solar wind splits; adjust PREPROC_SPLIT_* windows.")

    # Save each partition as a separate table in the split database
    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    print(f"[OK] IMF + solar wind imputed train/val/test splits stored at {OUTPUT_DB}")
    return train, val, test


def main() -> None:
    create_imf_solar_wind_splits()


if __name__ == "__main__":
    main()
