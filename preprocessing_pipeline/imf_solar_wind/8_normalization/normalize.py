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
    / "imf_solar_wind"
    / "7_train_test_split"
    / "imf_solar_wind_aver_comb_filt_imp_eng_split.db"
)

TRAIN_TABLE = "imf_solar_wind_train"
VAL_TABLE = "imf_solar_wind_validation"
TEST_TABLE = "imf_solar_wind_test"

OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb_filt_imp_eng_split_norm.db"
FINAL_COPY = STAGE_DIR.parents[1] / "imf_solar_wind" / "imf_solar_wind_fin.db"
PARAMS_PATH = STAGE_DIR / "imf_solar_wind_normalization.json"

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
# Column classification
# ---------------------------------------------------------------------
def _continuous_columns(columns: list[str]) -> list[str]:
    excluded_suffixes = (
        "_source_id",
        "_missing_flag",
    )
    excluded_exact = {
        "southward_flag",
        "high_speed_flag",
    }

    out: list[str] = []
    for col in columns:
        if col in excluded_exact:
            continue
        if col.endswith(excluded_suffixes):
            continue
        if col == "time_tag":
            continue
        out.append(col)
    return out


# ---------------------------------------------------------------------
# Normalization parameter fitting
# ---------------------------------------------------------------------
def _fit_z(series: pd.Series) -> dict[str, float]:
    mean = float(series.mean())
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    if not np.isfinite(mean):
        mean = 0.0
    return {"mean": mean, "std": std}


def fit_normalization_params(train: pd.DataFrame) -> dict[str, dict]:
    params: dict[str, dict] = {}
    cont_cols = _continuous_columns(list(train.columns))

    for col in cont_cols:
        series = train[col]
        if series.isna().any():
            continue
        params[col] = _fit_z(train[col])

    return params


# ---------------------------------------------------------------------
# Normalization application
# ---------------------------------------------------------------------
def apply_normalization(df: pd.DataFrame, params: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()

    for col, p in params.items():
        out[col] = (out[col] - p["mean"]) / p["std"]

    norm_cols = list(params.keys())
    if norm_cols:
        if out[norm_cols].isna().any().any():
            raise RuntimeError("NaNs introduced during normalization.")

    return out


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def normalize_sw_imf_splits() -> dict[str, pd.DataFrame]:
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
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="time_tag")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="time_tag")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="time_tag")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized IMF/SW splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters written to {PARAMS_PATH}")
    print(f"[OK] Copied final dataset to {FINAL_COPY}")

    return {
        "train": norm_train,
        "val": norm_val,
        "test": norm_test,
    }


def main() -> None:
    normalize_sw_imf_splits()


if __name__ == "__main__":
    main()
