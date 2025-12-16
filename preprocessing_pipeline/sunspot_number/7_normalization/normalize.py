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
# Paths and tables
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "sunspot_number"
    / "6_train_test_split"
    / "sunspot_number_aver_filt_imp_eng_split.db"
)

TRAIN_TABLE = "sunspot_train"
VAL_TABLE = "sunspot_validation"
TEST_TABLE = "sunspot_test"

OUTPUT_DB = STAGE_DIR / "sunspot_number_aver_filt_imp_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "sunspot_number_normalization.json"

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
# Normalization parameter fitting
# ---------------------------------------------------------------------
def _fit_zscore(series: pd.Series) -> dict[str, float]:
    mean = float(series.mean())
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        raise RuntimeError("Invalid z-score std encountered.")
    return {"mean": mean, "std": std}


def _fit_robust(series: pd.Series) -> dict[str, float]:
    median = float(series.median())
    iqr = float(series.quantile(0.75) - series.quantile(0.25))
    if not np.isfinite(iqr) or iqr == 0.0:
        raise RuntimeError("Invalid robust IQR encountered.")
    return {"median": median, "iqr": iqr}


def fit_normalization_params(train: pd.DataFrame) -> dict:
    params: dict[str, dict] = {}

    params["ssn_log"] = _fit_zscore(train["ssn_log"])
    params["ssn_mean_81d"] = _fit_zscore(train["ssn_mean_81d"])
    params["ssn_lag_81d"] = _fit_zscore(train["ssn_lag_81d"])
    params["ssn_slope_27d"] = _fit_robust(train["ssn_slope_27d"])

    log_persistence = np.log1p(train["ssn_persistence"])
    params["ssn_persistence"] = _fit_zscore(log_persistence)

    return params


# ---------------------------------------------------------------------
# Normalization application
# ---------------------------------------------------------------------
def _apply_z(series: pd.Series, p: dict) -> pd.Series:
    return (series - p["mean"]) / p["std"]


def _apply_robust(series: pd.Series, p: dict) -> pd.Series:
    return (series - p["median"]) / p["iqr"]


def apply_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()

    out["ssn_log"] = _apply_z(df["ssn_log"], params["ssn_log"])
    out["ssn_mean_81d"] = _apply_z(df["ssn_mean_81d"], params["ssn_mean_81d"])
    out["ssn_lag_81d"] = _apply_z(df["ssn_lag_81d"], params["ssn_lag_81d"])
    out["ssn_slope_27d"] = _apply_robust(df["ssn_slope_27d"], params["ssn_slope_27d"])

    out["ssn_cycle_phase"] = df["ssn_cycle_phase"]
    out["ssn_anomaly_cycle"] = df["ssn_anomaly_cycle"]

    out["ssn_persistence"] = _apply_z(
        np.log1p(df["ssn_persistence"]),
        params["ssn_persistence"],
    )

    return out


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def normalize_sunspot_splits() -> dict[str, pd.DataFrame]:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more data splits are empty.")

    params = fit_normalization_params(train)

    norm_train = apply_normalization(train, params)
    norm_val = apply_normalization(val, params)
    norm_test = apply_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))

    print(f"[OK] Normalized splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters written to {PARAMS_PATH}")

    return {
        "train": norm_train,
        "val": norm_val,
        "test": norm_test,
    }


def main() -> None:
    normalize_sunspot_splits()


if __name__ == "__main__":
    main()
