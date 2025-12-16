from __future__ import annotations

import sys
from pathlib import Path
import json
import sqlite3
from typing import Dict

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
    / "kp_index"
    / "6_train_test_split"
    / "kp_index_aver_filt_imp_eng_split.db"
)

TRAIN_TABLE = "kp_train"
VAL_TABLE = "kp_validation"
TEST_TABLE = "kp_test"

OUTPUT_DB = STAGE_DIR / "kp_index_aver_filt_imp_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "kp_index_normalization.json"

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def _load_split(table: str) -> pd.DataFrame:
    with sqlite3.connect(SPLITS_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
            conn,
            parse_dates=["timestamp"],
        )
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------
# Fit normalization params on TRAIN ONLY
# ---------------------------------------------------------------------
def fit_kp_normalization_params(train: pd.DataFrame) -> dict:
    params = {}

    for col in ["ap", "ap_3h_change"]:
        mean = float(train[col].mean())
        std = float(train[col].std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        params[col] = {"mean": mean, "std": std}

    return params


# ---------------------------------------------------------------------
# Apply normalization
# ---------------------------------------------------------------------
def apply_kp_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()

    # Continuous features
    for col in ["ap", "ap_3h_change"]:
        p = params[col]
        out[col] = (df[col] - p["mean"]) / p["std"]

    # Ordinal / categorical features: untouched
    out["kp_index"] = df["kp_index"]
    out["kp_regime"] = df["kp_regime"]
    out["ap_level_bucket"] = df["ap_level_bucket"]

    return out


# ---------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------
def normalize_kp_splits() -> Dict[str, pd.DataFrame]:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more KP splits are empty.")

    params = fit_kp_normalization_params(train)

    norm_train = apply_kp_normalization(train, params)
    norm_val = apply_kp_normalization(val, params)
    norm_test = apply_kp_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))

    print(f"[OK] Normalized KP splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters written to {PARAMS_PATH}")

    return {
        "train": norm_train,
        "val": norm_val,
        "test": norm_test,
    }


def main() -> None:
    normalize_kp_splits()


if __name__ == "__main__":
    main()
