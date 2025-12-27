from __future__ import annotations

import sys
from pathlib import Path
import sqlite3

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
    / "sunspot_number"
    / "5_engineered_features"
    / "sunspot_number_aver_filt_imp_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "sunspot_number_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"
MIN_ROWS_PER_WINDOW = 1   # SSN is quasi-static

# ---------------------------------------------------------------------
# Load sunspot features
# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("Sunspot engineered dataset is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected sunspot features indexed by timestamp.")
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
    rows = []

    grouped = df.groupby(
        pd.Grouper(freq=AGG_FREQ, label="right", closed="right")
    )

    for window_end, window in grouped:
        if len(window) < MIN_ROWS_PER_WINDOW:
            continue

        window = window.sort_index()
        last = window.iloc[-1]

        ssn_raw = last.get("ssn_raw")
        if not np.isfinite(ssn_raw):
            continue

        ssn_raw = float(ssn_raw)

        row = {
            "timestamp": window_end,

            # Absolute activity level
            "ssn_raw": ssn_raw,
            "ssn_log": float(np.log1p(ssn_raw)),

            # Long-term context (unchanged at 8h scale)
            "ssn_mean_81d": float(last.get("ssn_mean_81d", np.nan)),
            "ssn_slope_27d": float(last.get("ssn_slope_27d", np.nan)),
            "ssn_cycle_phase": float(last.get("ssn_cycle_phase", np.nan)),
            "ssn_lag_81d": float(last.get("ssn_lag_81d", np.nan)),
            "ssn_anomaly_cycle": float(last.get("ssn_anomaly_cycle", np.nan)),
            "ssn_persistence": int(last.get("ssn_persistence", 0)),
        }

        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h sunspot features produced.")

    features = features.sort_values("timestamp").reset_index(drop=True)
    return features


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_8h_sunspot_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_8h(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h sunspot features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_8h_sunspot_features()


if __name__ == "__main__":
    main()

