from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
INFERENCE_MODE = os.environ.get("XRS_INFERENCE_MODE") == "1"
AGG_FREQ = os.environ.get("PREPROC_AGG_FREQ", "1D").replace("H", "h")

SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "xray_flux"
    / "7_train_test_split"
    / "xray_flux_agg_eng_split.db"
)

TRAIN_TABLE = "xray_flux_train"
VAL_TABLE = "xray_flux_validation"
TEST_TABLE = "xray_flux_test"

OUTPUT_DB = STAGE_DIR / "xray_flux_agg_eng_split_norm.db"
PARAMS_PATH = STAGE_DIR / "xray_flux_normalization.json"
FINAL_COPY = STAGE_DIR.parents[0] / "xray_flux_fin.db"

AGG_DB = (
    STAGE_DIR.parents[1]
    / "xray_flux"
    / "6_aggregate"
    / "xray_flux_agg_eng.db"
)
AGG_TABLE = "features_agg"
INFERENCE_OUTPUT_DB = STAGE_DIR / "xray_flux_agg_inference_norm.db"

# ---------------------------------------------------------------------------
# Column groupings
# ---------------------------------------------------------------------------
Z_COLS: Set[str] = {
    "log_xrsb",
    "log_xrsb_max_24h",
    "log_xrsb_mean_6h",
    "log_xrsb_std_6h",
    "log_xrsb_slope_6h",
    "xrs_hardness",
    "log_xrsa",
    "log_xrsa_max_24h",
    "log_xrsa_mean_6h",
    "log_xrsa_std_6h",
    "log_xrsa_slope_6h",
}

LOG_Z_COLS: Set[str] = {
    "peak_to_bg_24h_xrsb",
    "peak_to_bg_24h_xrsa",
}

LOG1P_Z_COLS: Set[str] = {
    "hrs_since_rapid_rise_xrsb",
    "hrs_since_rapid_rise_xrsa",
}

BINARY_COLS: Set[str] = {
    "flaring_flag_xrsb",
    "flaring_flag_xrsa",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _fit_params(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = {}

    numeric_cols = set(train.select_dtypes(include=[np.number]).columns)
    classified = Z_COLS | LOG_Z_COLS | LOG1P_Z_COLS | BINARY_COLS
    missing_classification = numeric_cols - classified
    if missing_classification:
        raise RuntimeError(
            f"Unclassified numeric X-ray columns: {sorted(missing_classification)}"
        )

    for col in Z_COLS:
        if col in train.columns:
            params[col] = {"method": "z", **_fit_z(train[col])}

    for col in LOG_Z_COLS:
        if col in train.columns:
            transformed = np.log(
                _as_numeric(train[col]).clip(lower=np.finfo(float).eps)
            )
            params[col] = {"method": "log_z", **_fit_z(transformed)}

    for col in LOG1P_Z_COLS:
        if col in train.columns:
            transformed = np.log1p(
                _as_numeric(train[col]).clip(lower=0.0)
            )
            params[col] = {"method": "log1p_z", **_fit_z(transformed)}

    return params


def _apply_normalization(
    df: pd.DataFrame,
    params: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    out = df.copy()

    for col, info in params.items():
        if info["method"] == "z":
            series = _as_numeric(out[col])
            out[col] = (series - info["mean"]) / info["std"]
        elif info["method"] == "log_z":
            transformed = np.log(
                _as_numeric(out[col]).clip(lower=np.finfo(float).eps)
            )
            out[col] = (transformed - info["mean"]) / info["std"]
        elif info["method"] == "log1p_z":
            transformed = np.log1p(
                _as_numeric(out[col]).clip(lower=0.0)
            )
            out[col] = (transformed - info["mean"]) / info["std"]

    return out


def _load_params() -> Dict[str, Dict[str, float]]:
    if not PARAMS_PATH.exists():
        raise RuntimeError(
            f"Normalization parameter file missing: {PARAMS_PATH}. "
            "Run the full training pipeline to generate it."
        )
    return json.loads(PARAMS_PATH.read_text())


def _normalize_inference() -> None:
    params = _load_params()

    with sqlite3.connect(AGG_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {AGG_TABLE}", conn, parse_dates=["timestamp", "date"]
        )

    if df.empty:
        raise RuntimeError("X-ray aggregate dataset is empty.")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    else:
        raise RuntimeError("Aggregate dataset missing timestamp/date column.")
    if AGG_FREQ == "1D":
        df.index = df.index.normalize()

    normalized = _apply_normalization(df, params)

    with sqlite3.connect(INFERENCE_OUTPUT_DB) as conn:
        normalized.reset_index().to_sql(
            "normalized_features", conn, if_exists="replace", index=False
        )

    FINAL_COPY.write_bytes(INFERENCE_OUTPUT_DB.read_bytes())

    print(f"[OK] Applied saved normalization to {INFERENCE_OUTPUT_DB}")
    print(f"[OK] Copied inference-normalized dataset to {FINAL_COPY}")


def normalize_xray_flux_splits() -> None:
    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    if train.empty or val.empty or test.empty:
        raise RuntimeError("One or more X-ray splits are empty.")

    params = _fit_params(train)

    norm_train = _apply_normalization(train, params)
    norm_val = _apply_normalization(val, params)
    norm_test = _apply_normalization(test, params)

    with sqlite3.connect(OUTPUT_DB) as conn:
        norm_train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_val.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        norm_test.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    PARAMS_PATH.write_text(json.dumps(params, indent=2))

    FINAL_COPY.write_bytes(OUTPUT_DB.read_bytes())

    print(f"[OK] Normalized X-ray daily splits written to {OUTPUT_DB}")
    print(f"[OK] Normalization parameters saved to {PARAMS_PATH}")
    print(f"[OK] Copied normalized dataset to {FINAL_COPY}")


def main() -> None:
    if INFERENCE_MODE:
        _normalize_inference()
    else:
        normalize_xray_flux_splits()


if __name__ == "__main__":
    main()
