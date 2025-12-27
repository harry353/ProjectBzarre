from __future__ import annotations

import sys
from pathlib import Path
import json
import os
import sqlite3
from typing import Dict, Set

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
    / "sunspot_number"
    / "7_train_test_split"
    / "sunspot_number_agg_eng_split.db"
)

TRAIN_TABLE = "sunspot_train"
VAL_TABLE = "sunspot_validation"
TEST_TABLE = "sunspot_test"

OUTPUT_DB = STAGE_DIR / "sunspot_number_agg_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "sunspot_number_normalization.json"

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
    if AGG_FREQ == "1D":
        df.index = df.index.normalize()
    return df


# ---------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------
RAW_SSN: Set[str] = {
    "ssn_raw",
    "ssn_mean_81d",
    "ssn_cycle_phase",
}

Z_COLS: Set[str] = {
    "ssn_log",
    "ssn_anomaly_cycle",
}

ROBUST_COLS: Set[str] = {
    "ssn_slope_27d",
}

LOG_Z_COLS: Set[str] = {
    "ssn_persistence",
}

NO_SCALE_COLS = {
    "ssn_lag_81d",
}
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1D").replace("H", "h")

EXCLUDED_SUFFIXES = (
    "_source_id",
    "_missing_flag",
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fit_z(series: pd.Series) -> Dict[str, float]:
    s = _as_numeric(series).dropna()
    mean = float(s.mean()) if not s.empty else 0.0
    std = float(s.std()) if not s.empty else 1.0
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    if not np.isfinite(mean):
        mean = 0.0
    return {"mean": mean, "std": std}


def _fit_robust(series: pd.Series) -> Dict[str, float]:
    s = _as_numeric(series).dropna()
    median = float(s.median()) if not s.empty else 0.0
    iqr = float(s.quantile(0.75) - s.quantile(0.25)) if not s.empty else 1.0
    if not np.isfinite(iqr) or iqr == 0.0:
        iqr = 1.0
    return {"median": median, "iqr": iqr}


# ---------------------------------------------------------------------
# Fit normalization params (TRAIN ONLY)
# ---------------------------------------------------------------------
def fit_sunspot_normalization_params(train: pd.DataFrame) -> Dict[str, Dict]:
    params: Dict[str, Dict] = {}

    numeric_cols = set(train.select_dtypes(include=[np.number]).columns)
    classified = RAW_SSN | Z_COLS | ROBUST_COLS | LOG_Z_COLS | NO_SCALE_COLS
    unclassified = numeric_cols - classified
    if unclassified:
        raise RuntimeError(f"Unclassified numeric sunspot columns: {sorted(unclassified)}")

    for col in Z_COLS:
        if col in train.columns:
            params[col] = {"method": "z", **_fit_z(train[col])}

    for col in ROBUST_COLS:
        if col in train.columns:
            params[col] = {"method": "robust", **_fit_robust(train[col])}

    for col in LOG_Z_COLS:
        if col in train.columns:
            transformed = np.log1p(_as_numeric(train[col]).clip(lower=0.0))
            params[col] = {"method": "log_z", **_fit_z(transformed)}

    return params


# ---------------------------------------------------------------------
# Apply normalization
# ---------------------------------------------------------------------
def apply_sunspot_normalization(
    df: pd.DataFrame,
    params: Dict[str, Dict],
) -> pd.DataFrame:
    out = df.copy()

    for col, info in params.items():
        if info["method"] == "z":
            series = _as_numeric(out[col])
            out[col] = (series - info["mean"]) / info["std"]
        elif info["method"] == "robust":
            series = _as_numeric(out[col])
            out[col] = (series - info["median"]) / info["iqr"]
        elif info["method"] == "log_z":
            transformed = np.log1p(_as_numeric(out[col]).clip(lower=0.0))
            out[col] = (transformed - info["mean"]) / info["std"]

    return out


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def normalize_sunspot_splits() -> None:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more sunspot splits are empty.")

    params = fit_sunspot_normalization_params(train)

    norm_train = apply_sunspot_normalization(train, params)
    norm_val = apply_sunspot_normalization(val, params)
    norm_test = apply_sunspot_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))

    print(f"[OK] Normalized sunspot splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters written to {PARAMS_PATH}")


def main() -> None:
    normalize_sunspot_splits()


if __name__ == "__main__":
    main()
