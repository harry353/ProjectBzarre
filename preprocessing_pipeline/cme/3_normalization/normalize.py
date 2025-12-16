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
    / "cme"
    / "2_train_test_split"
    / "cme_hourly_eng_split.db"
)

TRAIN_TABLE = "cme_train"
VAL_TABLE = "cme_validation"
TEST_TABLE = "cme_test"

OUTPUT_DB = STAGE_DIR / "cme_hourly_eng_split_norm.db"
FINAL_COPY = STAGE_DIR.parents[1] / "cme" / "cme_fin.db"
PARAMS_PATH = STAGE_DIR / "cme_normalization.json"

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
# Column classification / rules
# ---------------------------------------------------------------------
LOG_Z_COLS = [
    "hours_since_last_cme",
    "hours_until_next_cme",
    "cme_count_last_24h",
    "cme_count_last_72h",
    "cme_strength_sum_24h",
    "cme_strength_sum_72h",
    "last_cme_v_med",
    "prev_cme_v_med",
    "last_cme_strength",
    "last_cme_shock_proxy",
    "cme_arrival_est_hours",
]

SIGNED_LOG_COLS = [
    "hours_until_est_arrival",
]

Z_COLS = [
    "last_cme_width",
    "effective_width",
]

ROBUST_COLS = [
    "last_cme_speed_ratio",
    "delta_last_cme_speed",
    "delta_last_cme_width",
]

def _safe_std(series: pd.Series) -> float:
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        return 1.0
    return std


def _log1p(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    values = np.clip(values, a_min=-0.999999, a_max=None)
    return np.log1p(values)


def _signed_log1p(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    return np.sign(values) * np.log1p(np.abs(values))


def fit_normalization_params(train: pd.DataFrame) -> dict[str, dict]:
    params: dict[str, dict] = {}

    def _maybe_add(col: str, method: str, transformed: pd.Series) -> None:
        if transformed.dropna().empty:
            return
        if method in {"log_z", "signed_log_z", "z"}:
            params[col] = {
                "method": method,
                "mean": float(transformed.mean()),
                "std": _safe_std(transformed),
            }
        elif method == "robust":
            median = float(transformed.median())
            q1 = float(transformed.quantile(0.25))
            q3 = float(transformed.quantile(0.75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0.0:
                iqr = 1.0
            params[col] = {
                "method": method,
                "median": median,
                "iqr": iqr,
            }

    for col in LOG_Z_COLS:
        if col in train.columns:
            transformed = _log1p(train[col])
            _maybe_add(col, "log_z", transformed)

    for col in SIGNED_LOG_COLS:
        if col in train.columns:
            transformed = _signed_log1p(train[col])
            _maybe_add(col, "signed_log_z", transformed)

    for col in Z_COLS:
        if col in train.columns:
            transformed = train[col].astype(float)
            _maybe_add(col, "z", transformed)

    for col in ROBUST_COLS:
        if col in train.columns:
            transformed = train[col].astype(float)
            _maybe_add(col, "robust", transformed)

    return params


# ---------------------------------------------------------------------
# Normalization application
# ---------------------------------------------------------------------
def apply_normalization(df: pd.DataFrame, params: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()

    for col, info in params.items():
        if col not in out.columns:
            continue
        method = info["method"]
        if method == "log_z":
            transformed = _log1p(out[col])
            out[col] = (transformed - info["mean"]) / info["std"]
        elif method == "signed_log_z":
            transformed = _signed_log1p(out[col])
            out[col] = (transformed - info["mean"]) / info["std"]
        elif method == "z":
            transformed = out[col].astype(float)
            out[col] = (transformed - info["mean"]) / info["std"]
        elif method == "robust":
            transformed = out[col].astype(float)
            out[col] = (transformed - info["median"]) / info["iqr"]

    return out


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def normalize_cme_splits() -> dict[str, pd.DataFrame]:
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
    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized CME splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters written to {PARAMS_PATH}")
    print(f"[OK] Copied final dataset to {FINAL_COPY}")

    return {
        "train": norm_train,
        "val": norm_val,
        "test": norm_test,
    }


def main() -> None:
    normalize_cme_splits()


if __name__ == "__main__":
    main()
