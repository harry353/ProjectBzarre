from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict

import numpy as np
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

STAGE_DIR = Path(__file__).resolve().parent
SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "dst"
    / "7_train_test_split"
    / "dst_agg_eng_split.db"
)
TRAIN_TABLE = "dst_train"
VAL_TABLE = "dst_validation"
TEST_TABLE = "dst_test"

OUTPUT_DB = STAGE_DIR / "dst_agg_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "dst_normalization.json"
FINAL_COPY = STAGE_DIR.parent / "dst_fin.db"

Z_SCORE_COLS = [
    "dst_std_8h",
    "dst_delta_8h",
    "dst_derivative",
]

NO_SCALE_COLS = [
    "dst",
    "dst_min_8h",
    "dst_mean_8h",
    "dst_recovery_flag",
]
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1D").replace("H", "h")


def _load_split(table: str) -> pd.DataFrame:
    with sqlite3.connect(SPLITS_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
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
        raise RuntimeError(f"Split {table} missing timestamp/date column.")
    if AGG_FREQ == "1D":
        df.index = df.index.normalize()
    return df


def _fit_params(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}
    missing = [col for col in Z_SCORE_COLS if col not in train.columns]
    if missing:
        raise RuntimeError(f"Expected columns missing in train split: {missing}")

    for col in Z_SCORE_COLS:
        series = train[col].dropna()
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        if not np.isfinite(mean):
            mean = 0.0
        params[col] = {"mean": mean, "std": std}
    return params


def _apply_normalization(df: pd.DataFrame, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, p in params.items():
        out[col] = (out[col] - p["mean"]) / p["std"]
    if params:
        norm_cols = list(params.keys())
        if out[norm_cols].isna().any().any():
            raise RuntimeError("NaNs introduced during DST normalization.")
        if (~np.isfinite(out[norm_cols])).any().any():
            raise RuntimeError("Infs introduced during DST normalization.")
    return out


def normalize_dst_daily():
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more DST splits are empty.")

    params = _fit_params(train)

    norm_train = _apply_normalization(train, params)
    norm_val = _apply_normalization(val, params)
    norm_test = _apply_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized DST daily splits written to {OUTPUT_DB}")
    print(f"[OK] Stored normalization parameters at {PARAMS_PATH}")
    print(f"[OK] Copied DST final dataset to {FINAL_COPY}")


def main() -> None:
    normalize_dst_daily()


if __name__ == "__main__":
    main()
