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
    / "imf_solar_wind"
    / "8_train_test_split"
    / "imf_solar_wind_agg_eng_split.db"
)

TRAIN_TABLE = "imf_solar_wind_train"
VAL_TABLE = "imf_solar_wind_validation"
TEST_TABLE = "imf_solar_wind_test"

OUTPUT_DB = STAGE_DIR / "imf_solar_wind_agg_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "imf_solar_wind_normalization.json"
FINAL_COPY = STAGE_DIR.parents[1] / "imf_solar_wind" / "imf_solar_wind_fin.db"

LOG_Z_COLS = [
    "density_mean_8h",
    "vbs_int_6h",
    "newell_dphi_dt_int_6h",
    "epsilon_max_6h",
    "pdyn_max_3h",
]

Z_COLS = [
    "speed_mean_8h",
    "hours_bz_south_last_8h",
    "bz_turning_rate_3h",
    "bt_mean_6h",
    "bz_std_6h",
    "delta_speed",
]

NO_SCALE_COLS = [
    "bz_min_24h",
    "southward_flag",
    "high_speed_flag",
]

LOG_CLIP = 1e-6
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

    missing = [col for col in LOG_Z_COLS + Z_COLS if col not in train.columns]
    if missing:
        raise RuntimeError(f"Missing expected IMF/SW aggregate columns: {missing}")

    for col in LOG_Z_COLS:
        transformed = np.log(_as_numeric(train[col]).clip(lower=LOG_CLIP))
        params[col] = {"method": "log_z", **_fit_z(transformed)}

    for col in Z_COLS:
        series = _as_numeric(train[col])
        params[col] = {"method": "z", **_fit_z(series)}

    return params


def apply_normalization(df: pd.DataFrame, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()

    for col, p in params.items():
        if p["method"] == "log_z":
            transformed = np.log(_as_numeric(out[col]).clip(lower=LOG_CLIP))
            out[col] = (transformed - p["mean"]) / p["std"]
        elif p["method"] == "z":
            series = _as_numeric(out[col])
            out[col] = (series - p["mean"]) / p["std"]

    for col in params.keys():
        mask = out[col].notna()
        if (~np.isfinite(out.loc[mask, col])).any():
            raise RuntimeError(f"Infinite values produced in column '{col}' during normalization.")

    return out


def normalize_imf_solar_wind_splits() -> None:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more IMF/SW splits are empty.")

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

    print(f"[OK] Normalized IMF/SW aggregate splits written to {OUTPUT_DB}")
    print(f"[OK] Stored normalization parameters at {PARAMS_PATH}")
    print(f"[OK] Copied final dataset to {FINAL_COPY}")


def main() -> None:
    normalize_imf_solar_wind_splits()


if __name__ == "__main__":
    main()
