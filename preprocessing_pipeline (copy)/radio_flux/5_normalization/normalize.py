from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root resolution
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
    / "radio_flux"
    / "4_train_test_split"
    / "radio_flux_agg_eng_split.db"
)

TRAIN_TABLE = "radio_flux_train"
VAL_TABLE = "radio_flux_validation"
TEST_TABLE = "radio_flux_test"

OUTPUT_DB = STAGE_DIR / "radio_flux_agg_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "radio_flux_normalization.json"
FINAL_COPY = STAGE_DIR.parents[1] / "radio_flux" / "radio_flux_fin.db"

# ---------------------------------------------------------------------
# Column groups (AUTHORITATIVE)
# ---------------------------------------------------------------------
Z_COLS = [
    "log_f107",
    "f107_mean_24h",
    "f107_delta_24h",
    "f107_std_72h",
    "log_f107_mean_24h",
    "f107_anomaly_72h",
]

NO_SCALE_COLS = [
    "f107",
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_split(table: str) -> pd.DataFrame:
    with sqlite3.connect(SPLITS_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
            conn,
            parse_dates=["timestamp"],
        )

    if "timestamp" not in df.columns:
        raise RuntimeError(f"Split {table} missing timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp").sort_index()

    return df


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


def fit_normalization_params(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}

    missing = [c for c in Z_COLS + NO_SCALE_COLS if c not in train.columns]
    if missing:
        raise RuntimeError(f"Missing expected radio flux columns: {missing}")

    for col in Z_COLS:
        params[col] = {
            "method": "z",
            **_fit_z(train[col]),
        }

    return params


def apply_normalization(
    df: pd.DataFrame,
    params: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    out = df.copy()

    for col, p in params.items():
        series = _as_numeric(out[col])
        out[col] = (series - p["mean"]) / p["std"]

    norm_cols = list(params.keys())
    if norm_cols:
        if out[norm_cols].isna().any().any():
            raise RuntimeError("NaNs introduced during radio flux normalization.")
        if (~np.isfinite(out[norm_cols])).any().any():
            raise RuntimeError("Infs introduced during radio flux normalization.")

    return out


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def normalize_radio_flux_splits() -> None:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more radio flux splits are empty.")

    params = fit_normalization_params(train)

    norm_train = apply_normalization(train, params)
    norm_val = apply_normalization(val, params)
    norm_test = apply_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(
            TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp"
        )
        norm_val.to_sql(
            VAL_TABLE, conn, if_exists="replace", index_label="timestamp"
        )
        norm_test.to_sql(
            TEST_TABLE, conn, if_exists="replace", index_label="timestamp"
        )

    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized radio flux splits written to {OUTPUT_DB}")
    print(f"[OK] Stored normalization parameters at {PARAMS_PATH}")
    print(f"[OK] Copied final radio flux DB to {FINAL_COPY}")


def main() -> None:
    normalize_radio_flux_splits()


if __name__ == "__main__":
    main()
