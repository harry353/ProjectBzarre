from __future__ import annotations

import sys
from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project path resolution
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "flares"
    / "5_train_test_split"
    / "flare_agg_eng_split.db"
)

TRAIN_TABLE = "flare_train"
VAL_TABLE = "flare_validation"
TEST_TABLE = "flare_test"

OUTPUT_DB = STAGE_DIR / "flare_agg_eng_split_norm.db"
FINAL_COPY = STAGE_DIR.parents[1] / "flares" / "flare_fin.db"
PARAMS_PATH = STAGE_DIR / "flare_normalization.json"

BINARY_COLS = ["flare_active_flag", "flare_major_flag", "flare_extreme_flag"]
ORDINAL_COLS = ["flare_class_ord"]
LOG1P_COLS = [
    "hours_since_last_flare",
    "flare_count_last_24h",
    "flare_count_last_72h",
    "flare_influence_exp",
]
LOG_POS_COLS = [
    "last_flare_peak_flux",
    "last_flare_integrated_flux",
]

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
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
    df.index = df.index.normalize()
    return df


# ---------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------
def _fit_z(series: pd.Series) -> dict:
    mean = float(series.mean())
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    if not np.isfinite(mean):
        mean = 0.0
    return {"mean": mean, "std": std}


# ---------------------------------------------------------------------
# Fit normalization parameters (TRAIN ONLY)
# ---------------------------------------------------------------------
def fit_flare_normalization_params(train: pd.DataFrame) -> dict:
    params: dict[str, dict] = {}

    missing = [col for col in LOG1P_COLS + LOG_POS_COLS if col not in train.columns]
    if missing:
        raise RuntimeError(f"Missing columns in flare train split: {missing}")

    for col in LOG1P_COLS:
        params[col] = _fit_z(np.log1p(np.maximum(train[col], 0.0)))

    for col in LOG_POS_COLS:
        params[col] = _fit_z(np.log(np.maximum(train[col], 1e-12)))

    return params


# ---------------------------------------------------------------------
# Apply normalization
# ---------------------------------------------------------------------
def apply_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()

    for col in LOG1P_COLS:
        out[col] = (np.log1p(np.maximum(df[col], 0.0)) - params[col]["mean"]) / params[col]["std"]

    for col in LOG_POS_COLS:
        out[col] = (np.log(np.maximum(df[col], 1e-12)) - params[col]["mean"]) / params[col]["std"]

    # ordinal and binary columns remain unchanged
    return out


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def normalize_flare_splits() -> dict[str, pd.DataFrame]:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more flare splits are empty.")

    params = fit_flare_normalization_params(train)

    norm_train = apply_normalization(train, params)
    norm_val = apply_normalization(val, params)
    norm_test = apply_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized flare splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters written to {PARAMS_PATH}")
    print(f"[OK] Copied final dataset to {FINAL_COPY}")

    return {
        "train": norm_train,
        "val": norm_val,
        "test": norm_test,
    }


def main() -> None:
    normalize_flare_splits()


if __name__ == "__main__":
    main()
