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
    / "4_train_test_split"
    / "flare_comb_filt_eng_split.db"
)

TRAIN_TABLE = "flare_train"
VAL_TABLE = "flare_validation"
TEST_TABLE = "flare_test"

OUTPUT_DB = STAGE_DIR / "flare_comb_filt_eng_split_norm.db"
FINAL_COPY = STAGE_DIR.parents[1] / "flares" / "flare_fin.db"
PARAMS_PATH = STAGE_DIR / "flare_normalization.json"

# ---------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------
BINARY_COLS = [
    "flare_active_flag",
    "flare_major_flag",
    "flare_extreme_flag",
    "flare_overtaking_flag",
]

ORDINAL_COLS = [
    "flare_class_ord",
]

NO_SCALE_COLS = [
    "flare_influence_exp",
]

LOG_TIME_COLS = [
    "hours_since_last_flare",
    "hours_until_next_flare",
]

COUNT_COLS = [
    "flare_count_last_6h",
    "flare_count_last_24h",
    "flare_count_last_72h",
]

LOG_FLUX_COLS = [
    "last_flare_peak_flux",
    "last_flare_integrated_flux",
    "last_flare_background_flux",
    "last_flare_xrsb_flux",
    "last_flare_energy_proxy",
]

ROBUST_COLS = [
    "delta_last_flare_peak_flux",
    "flare_energy_ratio",
]

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
    df.index = pd.to_datetime(df.index, utc=True)
    return df


# ---------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------
def _fit_z(series: pd.Series) -> dict:
    mean = float(series.mean())
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        raise RuntimeError("Invalid std during z-score fitting.")
    return {"mean": mean, "std": std}


def _fit_robust(series: pd.Series) -> dict:
    median = float(series.median())
    iqr = float(series.quantile(0.75) - series.quantile(0.25))
    if not np.isfinite(iqr) or iqr == 0.0:
        raise RuntimeError("Invalid IQR during robust fitting.")
    return {"median": median, "iqr": iqr}


# ---------------------------------------------------------------------
# Fit normalization parameters (TRAIN ONLY)
# ---------------------------------------------------------------------
def fit_flare_normalization_params(train: pd.DataFrame) -> dict:
    params: dict[str, dict] = {}

    for col in LOG_TIME_COLS + COUNT_COLS:
        params[col] = _fit_z(np.log1p(train[col]))

    for col in LOG_FLUX_COLS:
        params[col] = _fit_z(np.log10(np.maximum(train[col], 1e-20)))

    for col in ROBUST_COLS:
        params[col] = _fit_robust(train[col])

    return params


# ---------------------------------------------------------------------
# Apply normalization
# ---------------------------------------------------------------------
def apply_normalization(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()

    for col in LOG_TIME_COLS + COUNT_COLS:
        out[col] = (np.log1p(df[col]) - params[col]["mean"]) / params[col]["std"]

    for col in LOG_FLUX_COLS:
        out[col] = (
            np.log10(np.maximum(df[col], 1e-20)) - params[col]["mean"]
        ) / params[col]["std"]

    for col in ROBUST_COLS:
        out[col] = (df[col] - params[col]["median"]) / params[col]["iqr"]

    # Ordinal: optional scaling
    for col in ORDINAL_COLS:
        out[col] = df[col] / 4.0

    # Binary + no-scale columns untouched
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
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="time_tag")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="time_tag")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="time_tag")

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
