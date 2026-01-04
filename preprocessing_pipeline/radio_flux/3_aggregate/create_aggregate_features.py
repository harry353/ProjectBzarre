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
    / "radio_flux"
    / "2_engineered_features"
    / "radio_flux_filt_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "radio_flux_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Aggregation parameters (AUTHORITATIVE)
# ---------------------------------------------------------------------
MEAN_WINDOW_H = 24
DELTA_WINDOW_H = 24
STD_WINDOW_H = 72
LOG_MEAN_WINDOW_H = 24
ANOMALY_WINDOW_H = 72

MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Load hourly radio flux features
# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("Radio flux engineered dataset is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Expected radio flux indexed by timestamp.")

    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    required = {
        "f107",
        "log_f107",
        "df107_24h",
        "f107_anomaly_27d",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required engineered features: {sorted(missing)}")

    return df


# ---------------------------------------------------------------------
# Build rolling aggregates (PAST-ONLY, HOURLY CADENCE)
# ---------------------------------------------------------------------
def _build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _min_periods(w: int) -> int:
        return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))

    # --------------------------------------------------------------
    # Smoothed background
    # --------------------------------------------------------------
    w = MEAN_WINDOW_H
    out[f"f107_mean_{w}h"] = (
        df["f107"]
        .rolling(w, min_periods=_min_periods(w))
        .mean()
    )

    # --------------------------------------------------------------
    # Trend over window
    # --------------------------------------------------------------
    w = DELTA_WINDOW_H
    out[f"f107_delta_{w}h"] = df["f107"] - df["f107"].shift(w)

    # --------------------------------------------------------------
    # Variability
    # --------------------------------------------------------------
    w = STD_WINDOW_H
    out[f"f107_std_{w}h"] = (
        df["f107"]
        .rolling(w, min_periods=_min_periods(w))
        .std(ddof=0)
    )

    # --------------------------------------------------------------
    # Log-space baseline
    # --------------------------------------------------------------
    w = LOG_MEAN_WINDOW_H
    out[f"log_f107_mean_{w}h"] = (
        df["log_f107"]
        .rolling(w, min_periods=_min_periods(w))
        .mean()
    )

    # --------------------------------------------------------------
    # Contextual anomaly
    # --------------------------------------------------------------
    w = ANOMALY_WINDOW_H
    out[f"f107_anomaly_{w}h"] = (
        df["f107"]
        - df["f107"]
        .rolling(w, min_periods=_min_periods(w))
        .mean()
    )

    out = out.dropna()

    if out.empty:
        raise RuntimeError("No radio flux aggregate features produced.")

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_radio_flux_agg_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_aggregates(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] Radio flux aggregate features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_radio_flux_agg_features()


if __name__ == "__main__":
    main()

