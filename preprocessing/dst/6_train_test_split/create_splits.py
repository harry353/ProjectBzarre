from __future__ import annotations
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

from preprocessing.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent
FEATURES_DB = STAGE_DIR.parents[1] / "dst" / "5_engineered_features" / "dst_eng.db"
FEATURES_TABLE = "engineered_features"
OUTPUT_DB = STAGE_DIR / "dst_split.db"
TRAIN_TABLE = "dst_train"
VAL_TABLE = "dst_validation"
TEST_TABLE = "dst_test"

TRAIN_START = pd.Timestamp("2005-01-01T00:00:00Z")
VAL_START = pd.Timestamp("2016-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2018-01-01T00:00:00Z")


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


def create_dst_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("DST features dataset is empty; run feature engineering first.")
    df = _prepare_index(df)

    train = df.loc[(df.index >= TRAIN_START) & (df.index < VAL_START)]
    val = df.loc[(df.index >= VAL_START) & (df.index < TEST_START)]
    test = df.loc[df.index >= TEST_START]

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more DST splits are empty; verify time coverage.")

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
