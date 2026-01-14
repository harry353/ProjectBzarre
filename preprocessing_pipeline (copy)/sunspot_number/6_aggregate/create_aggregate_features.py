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
# Aggregation parameters (AUTHORITATIVE)
# ---------------------------------------------------------------------
MEAN_STD_WINDOW_H = 81 * 24
TREND_WINDOW_H = 54 * 24

MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Load hourly sunspot features
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
# Build aggregates (PAST-ONLY, HOURLY CADENCE)
# ---------------------------------------------------------------------
def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _min_periods(w: int) -> int:
        return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))

    # --------------------------------------------------------------
    # Background level and variability
    # --------------------------------------------------------------
    w = MEAN_STD_WINDOW_H
    window = f"{w}h"
    out[f"ssn_mean_{w}h"] = (
        df["ssn"]
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    out[f"ssn_std_{w}h"] = (
        df["ssn"]
        .rolling(window, min_periods=_min_periods(w))
        .std()
        .fillna(0.0)
    )

    # --------------------------------------------------------------
    # Medium-term trend confirmation
    # --------------------------------------------------------------
    w = TREND_WINDOW_H
    lagged = df["ssn"].shift(w)
    out[f"ssn_trend_{w}h"] = (df["ssn"] - lagged) / (w / 24.0)

    # --------------------------------------------------------------
    # Fraction of time above background
    # --------------------------------------------------------------
    mean_ref = out[f"ssn_mean_{MEAN_STD_WINDOW_H}h"]
    above = (df["ssn"] > mean_ref).astype(float)

    out[f"ssn_anomaly_frac_{MEAN_STD_WINDOW_H}h"] = (
        above
        .rolling(f"{MEAN_STD_WINDOW_H}h", min_periods=_min_periods(MEAN_STD_WINDOW_H))
        .mean()
    )

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    if out.empty:
        raise RuntimeError("No sunspot aggregate features produced.")

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_sunspot_agg_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_agg(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] Sunspot aggregate features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features

def main() -> None:
    create_sunspot_agg_features()

if __name__ == "__main__":
    main()
