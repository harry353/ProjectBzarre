from __future__ import annotations

import json
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
SPLITS_DB = STAGE_DIR.parents[1] / "cme" / "3_train_test_split" / "cme_agg_eng_split.db"
TRAIN_TABLE = "cme_train"
VAL_TABLE = "cme_validation"
TEST_TABLE = "cme_test"
OUTPUT_DB = STAGE_DIR / "cme_agg_eng_split_norm.db"
FINAL_COPY = STAGE_DIR.parents[1] / "cme" / "cme_fin.db"
PARAMS_PATH = STAGE_DIR / "cme_normalization.json"

LOG_Z_COLS = [
    # Engineered
    "hours_since_last_cme",
    "last_cme_v_med",
    "cme_strength_sum_24h",
    "last_cme_shock_proxy",

    # Aggregates
    "min_hours_since_last_cme_8h",
    "max_last_cme_v_med_8h",
    "max_last_cme_shock_proxy_8h",
]

Z_COLS = [
    # Engineered
    "effective_width",
    "cme_influence_exp",

    # Aggregates
    "mean_cme_influence_exp_8h",
]

NO_SCALE_COLS = [
    "cme_overtaking_flag",
]

ROBUST_COLS = [
    # keep empty or add columns that need robust scaling
]

def _load_split(table: str) -> pd.DataFrame:
    with sqlite3.connect(SPLITS_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
            conn,
            parse_dates=["timestamp", "date"],
        )
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.set_index("timestamp").sort_index()
    if "date" in df.columns:
        df = df.dropna(subset=["date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.set_index("date").sort_index()
    raise RuntimeError(f"Split {table} missing timestamp/date column.")


def _safe_std(series: pd.Series) -> float:
    std = float(series.std())
    if not np.isfinite(std) or std == 0.0:
        return 1.0
    return std


def _log1p(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    values = np.clip(values, a_min=-0.999999, a_max=None)
    return np.log1p(values)


def _fit_robust(series: pd.Series) -> Dict[str, float]:
    median = float(series.median())
    iqr = float(series.quantile(0.75) - series.quantile(0.25))
    if not np.isfinite(iqr) or iqr == 0.0:
        iqr = 1.0
    if not np.isfinite(median):
        median = 0.0
    return {"median": median, "iqr": iqr}


def fit_normalization_params(train: pd.DataFrame) -> dict[str, dict]:
    params: dict[str, dict] = {}

    classified = set(NO_SCALE_COLS)
    classified.update(LOG_Z_COLS)
    classified.update(Z_COLS)
    classified.update(ROBUST_COLS)

    numeric_cols = set(train.select_dtypes(include=[np.number]).columns)
    unclassified = numeric_cols - classified
    if unclassified:
        raise RuntimeError(f"Unclassified numeric CME columns: {sorted(unclassified)}")

    for col in LOG_Z_COLS:
        if col in train.columns:
            series = _log1p(train[col])
            if series.dropna().empty:
                continue
            params[col] = {
                "method": "log_z",
                "mean": float(series.mean()),
                "std": _safe_std(series),
            }

    for col in Z_COLS:
        if col in train.columns:
            series = train[col].astype(float)
            if series.dropna().empty:
                continue
            params[col] = {
                "method": "z",
                "mean": float(series.mean()),
                "std": _safe_std(series),
            }

    for col in ROBUST_COLS:
        if col in train.columns:
            series = train[col].astype(float)
            if series.dropna().empty:
                continue
            params[col] = {
                "method": "robust",
                **_fit_robust(series),
            }

    return params


def apply_normalization(df: pd.DataFrame, params: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    for col, info in params.items():
        if col not in out.columns:
            continue
        method = info["method"]
        if method == "log_z":
            transformed = _log1p(out[col])
            out[col] = (transformed - info["mean"]) / info["std"]
        elif method == "z":
            transformed = out[col].astype(float)
            out[col] = (transformed - info["mean"]) / info["std"]
        elif method == "robust":
            transformed = out[col].astype(float)
            out[col] = (transformed - info["median"]) / info["iqr"]
    return out


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

    return {"train": norm_train, "validation": norm_val, "test": norm_test}


def main() -> None:
    normalize_cme_splits()


if __name__ == "__main__":
    main()
