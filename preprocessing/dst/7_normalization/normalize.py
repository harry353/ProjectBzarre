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

import json
import sqlite3
from typing import Dict, List

import numpy as np
import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent
SPLITS_DB = STAGE_DIR.parents[1] / "dst" / "6_train_test_split" / "dst_split.db"
TRAIN_TABLE = "dst_train"
VAL_TABLE = "dst_validation"
TEST_TABLE = "dst_test"
OUTPUT_DB = STAGE_DIR / "dst_norm.db"
PARAMS_PATH = STAGE_DIR / "dst_normalization.json"


def _load_split(table: str) -> pd.DataFrame:
    with sqlite3.connect(SPLITS_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df


def _continuous_columns(columns: List[str]) -> List[str]:
    return [col for col in columns if col != "timestamp"]


def normalize_dst_splits() -> Dict[str, pd.DataFrame]:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more splits are empty; run the split step first.")

    columns = _continuous_columns(list(train.columns))
    means = train[columns].mean()
    stds = train[columns].std().replace(0.0, 1.0).fillna(1.0)

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        for col in columns:
            normalized[col] = (normalized[col] - means[col]) / stds[col]
        return normalized

    norm_train = _normalize(train)
    norm_val = _normalize(val)
    norm_test = _normalize(test)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    payload = {
        "columns": columns,
        "mean": means.to_dict(),
        "std": stds.to_dict(),
    }
    PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Normalized splits written to {OUTPUT_DB}")
    return {"train": norm_train, "val": norm_val, "test": norm_test}


def main() -> None:
    normalize_dst_splits()


if __name__ == "__main__":
    main()
