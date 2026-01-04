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
    / "cme"
    / "1_engineered_features"
    / "cme_hourly_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "cme_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ------------------------------------------------------------------
# Aggregation parameters (authoritative)
# ------------------------------------------------------------------

MIN_FRACTION_COVERAGE = 0.5

HOURS_SINCE_MIN_WINDOW_H = 6
V_MED_MAX_WINDOW_H = 12
INFLUENCE_MEAN_WINDOW_H = 12
SHOCK_MAX_WINDOW_H = 6

# ---------------------------------------------------------------------
# Load hourly CME features
# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("CME engineered dataset is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected CME features indexed by timestamps.")
    df = df.sort_index()
    df.index = df.index.tz_convert("UTC")
    return df

# ---------------------------------------------------------------------
# Build aggregates (PAST-ONLY, HOURLY CADENCE)
# ---------------------------------------------------------------------
def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    out = df.copy()

    def _min_periods(w: int) -> int:
        return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))

    # --------------------------------------------------------------
    # Closest CME in recent history
    # --------------------------------------------------------------
    w = HOURS_SINCE_MIN_WINDOW_H
    out[f"min_hours_since_last_cme_{w}h"] = (
        df["hours_since_last_cme"]
        .rolling(w, min_periods=_min_periods(w))
        .min()
    )

    # --------------------------------------------------------------
    # Fastest CME observed recently
    # --------------------------------------------------------------
    w = V_MED_MAX_WINDOW_H
    out[f"max_last_cme_v_med_{w}h"] = (
        df["last_cme_v_med"]
        .rolling(w, min_periods=_min_periods(w))
        .max()
    )

    # --------------------------------------------------------------
    # Sustained CME influence
    # --------------------------------------------------------------
    w = INFLUENCE_MEAN_WINDOW_H
    out[f"mean_cme_influence_exp_{w}h"] = (
        df["cme_influence_exp"]
        .rolling(w, min_periods=_min_periods(w))
        .mean()
    )

    # --------------------------------------------------------------
    # Strongest shock signature
    # --------------------------------------------------------------
    w = SHOCK_MAX_WINDOW_H
    out[f"max_last_cme_shock_proxy_{w}h"] = (
        df["last_cme_shock_proxy"]
        .rolling(w, min_periods=_min_periods(w))
        .max()
    )

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------
    out = out.dropna()

    if out.empty:
        raise RuntimeError("No aggregated CME features produced.")

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_cme_agg_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_agg(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] CME aggregate features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_cme_agg_features()


if __name__ == "__main__":
    main()
