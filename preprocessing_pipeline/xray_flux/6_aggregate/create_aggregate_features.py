from __future__ import annotations

import sys
from pathlib import Path
import sqlite3
from typing import Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project paths
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

from preprocessing_pipeline.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent

FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "xray_flux"
    / "5_feature_engineering"
    / "xray_flux_aver_filt_imp_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "xray_flux_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"
MIN_ROWS_PER_WINDOW = 4   # require ~half coverage

REQUIRED_COLS = [
    "log_xrsb",
    "log_xrsb_mean_6h",
    "log_xrsb_std_6h",
    "log_xrsb_slope_6h",
    "peak_to_bg_24h_xrsb",
    "hrs_since_rapid_rise_xrsb",
    "flaring_flag_xrsb",
    "xrs_hardness",
    "log_xrsa",
    "log_xrsa_mean_6h",
    "log_xrsa_std_6h",
    "log_xrsa_slope_6h",
    "peak_to_bg_24h_xrsa",
    "hrs_since_rapid_rise_xrsa",
    "flaring_flag_xrsa",
]

# ---------------------------------------------------------------------
# Load hourly features
# ---------------------------------------------------------------------
def _load_hourly_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("X-ray engineered dataset is empty.")

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Required column '{col}' missing.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected X-ray features indexed by timestamp.")

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df


# ---------------------------------------------------------------------
# Build 8h aggregates (PAST-ONLY)
# ---------------------------------------------------------------------
def _build_8h(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []

    grouped = df.groupby(
        pd.Grouper(freq=AGG_FREQ, label="right", closed="right")
    )

    for window_end, window in grouped:
        if len(window) < MIN_ROWS_PER_WINDOW:
            continue

        window = window.sort_index()
        last = window.iloc[-1]

        row = {
            "timestamp": window_end,

            # GOES XRS-B (primary flare proxy)
            "log_xrsb": float(last["log_xrsb"]),
            "log_xrsb_mean_6h": float(last["log_xrsb_mean_6h"]),
            "log_xrsb_std_6h": float(last["log_xrsb_std_6h"]),
            "log_xrsb_slope_6h": float(last["log_xrsb_slope_6h"]),
            "peak_to_bg_24h_xrsb": float(last["peak_to_bg_24h_xrsb"]),
            "hrs_since_rapid_rise_xrsb": float(last["hrs_since_rapid_rise_xrsb"]),
            "flaring_flag_xrsb": int(last["flaring_flag_xrsb"]),

            # Spectral information
            "xrs_hardness": float(last["xrs_hardness"]),

            # GOES XRS-A (contextual)
            "log_xrsa": float(last["log_xrsa"]),
            "log_xrsa_mean_6h": float(last["log_xrsa_mean_6h"]),
            "log_xrsa_std_6h": float(last["log_xrsa_std_6h"]),
            "log_xrsa_slope_6h": float(last["log_xrsa_slope_6h"]),
            "peak_to_bg_24h_xrsa": float(last["peak_to_bg_24h_xrsa"]),
            "hrs_since_rapid_rise_xrsa": float(last["hrs_since_rapid_rise_xrsa"]),
            "flaring_flag_xrsa": int(last["flaring_flag_xrsa"]),
        }

        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h X-ray feature rows produced.")

    features = features.sort_values("timestamp").reset_index(drop=True)
    return features


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_8h_xray_features() -> pd.DataFrame:
    df = _load_hourly_features()
    features = _build_8h(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h X-ray features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_8h_xray_features()


if __name__ == "__main__":
    main()

