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

from preprocessing_pipeline.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent
FEATURES_DB = STAGE_DIR.parents[1] / "dst" / "5_engineered_features" / "dst_aver_filt_imp_eng.db"
FEATURES_TABLE = "engineered_features"
OUTPUT_DB = STAGE_DIR / "dst_aver_filt_imp_eng_split.db"
TRAIN_TABLE = "dst_train"
VAL_TABLE = "dst_validation"
TEST_TABLE = "dst_test"

DEFAULT_WINDOWS = {
    "train": ("1999-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-11-30"),
}


def _prepare_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Feature dataframe must have a DateTimeIndex.")
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
    return df.loc[(df.index >= start) & (df.index <= end)]


def create_dst_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("DST features dataset is empty; run feature engineering first.")
    df = _prepare_index(df)

    windows = _get_windows()
    train = _slice(df, *windows["train"])
    val = _slice(df, *windows["validation"])
    test = _slice(df, *windows["test"])

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more DST splits are empty; adjust PREPROC_SPLIT_* env settings or confirm coverage.")

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    print(f"[OK] DST train/val/test splits stored at {OUTPUT_DB}")
    return train, val, test


def main() -> None:
    create_dst_splits()


if __name__ == "__main__":
    main()
