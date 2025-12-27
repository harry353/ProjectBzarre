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
    / "kp_index"
    / "7_train_test_split"
    / "kp_index_agg_eng_split.db"
)

TRAIN_TABLE = "kp_train"
VAL_TABLE = "kp_validation"
TEST_TABLE = "kp_test"

OUTPUT_DB = STAGE_DIR / "kp_index_agg_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "kp_index_normalization.json"
FINAL_COPY = STAGE_DIR.parent / "kp_fin.db"

LOG1P_COLS = [
    "ap_sum_8h",
    "ap_max_8h",
]

Z_COLS = [
    "kp_delta_6h",
    "kp_accel",
    "kp_hours_above_5",
]

NO_SCALE_COLS = [
    "kp_index",
    "kp_max_24h",
    "kp_mean_12h",
    "kp_dist_to_5",
    "kp_regime",
    "kp_jump_2plus",
]
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1D").replace("H", "h")


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


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


def _fit_z(series: pd.Series) -> Dict[str, float]:
    mean = float(series.mean())
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    if not np.isfinite(mean):
        mean = 0.0
    return {"mean": mean, "std": std}


def fit_normalization_params(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}

    missing = [col for col in LOG1P_COLS + Z_COLS if col not in train.columns]
    if missing:
        raise RuntimeError(f"Missing expected KP aggregate columns: {missing}")

    for col in LOG1P_COLS:
        transformed = np.log1p(_as_numeric(train[col]).clip(lower=0.0))
        params[col] = {"method": "log1p_z", **_fit_z(transformed)}

    for col in Z_COLS:
        series = _as_numeric(train[col])
        params[col] = {"method": "z", **_fit_z(series)}

    return params


def apply_normalization(df: pd.DataFrame, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()

    for col, p in params.items():
        if p["method"] == "log1p_z":
            transformed = np.log1p(_as_numeric(out[col]).clip(lower=0.0))
            out[col] = (transformed - p["mean"]) / p["std"]
        elif p["method"] == "z":
            series = _as_numeric(out[col])
            out[col] = (series - p["mean"]) / p["std"]

    norm_cols = list(params.keys())
    if norm_cols:
        if out[norm_cols].isna().any().any():
            raise RuntimeError("NaNs introduced during KP normalization.")
        if (~np.isfinite(out[norm_cols])).any().any():
            raise RuntimeError("Infs introduced during KP normalization.")

    return out


def normalize_kp_splits() -> None:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more KP splits are empty.")

    params = fit_normalization_params(train)

    norm_train = apply_normalization(train, params)
    norm_val = apply_normalization(val, params)
    norm_test = apply_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized KP daily splits written to {OUTPUT_DB}")
    print(f"[OK] Stored normalization parameters at {PARAMS_PATH}")
    print(f"[OK] Copied KP final dataset to {FINAL_COPY}")


def main() -> None:
    normalize_kp_splits()


if __name__ == "__main__":
    main()
