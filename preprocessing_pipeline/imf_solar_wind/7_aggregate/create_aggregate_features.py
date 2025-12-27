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
    / "imf_solar_wind"
    / "6_engineered_features"
    / "imf_solar_wind_aver_comb_filt_imp_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "imf_solar_wind_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"
MIN_ROWS_PER_WINDOW = 4      # require ≥50% coverage
HIGH_SPEED_THRESHOLD = 600.0

# ---------------------------------------------------------------------
# Load hourly IMF + solar wind features
# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("IMF + solar wind engineered dataset is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected IMF features indexed by timestamps.")
    df = df.sort_index()
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
        last3 = window.tail(3)
        last6 = window.tail(6)

        prev_speed = window["speed"].iloc[-2] if len(window) >= 2 else np.nan

        bz_diff = last3["bz_gse"].diff().abs()

        row = {
            "timestamp": window_end,

            # Plasma background
            "density_mean_8h": window["density"].mean(),
            "speed_mean_8h": window["speed"].mean(),

            # IMF orientation
            "bz_min_8h": window["bz_gse"].min(),
            "bz_std_6h": last6["bz_gse"].std(),

            # Southward persistence
            "hours_bz_south_last_8h": (window["bz_gse"] < 0).sum(),

            # Coupling proxies (recent)
            "vbs_int_6h": last6["vbs"].sum(),
            "newell_dphi_dt_int_6h": last6["newell_dphi_dt"].sum(),
            "epsilon_max_6h": last6["epsilon"].max(),

            # Dynamic pressure
            "pdyn_max_3h": last3["dynamic_pressure"].max(),

            # Variability
            "bz_turning_rate_3h": bz_diff.mean(),

            # Flags (current state)
            "southward_flag": int(last["bz_gse"] < 0),
            "high_speed_flag": int(last["speed"] > HIGH_SPEED_THRESHOLD),

            # Magnetic field strength
            "bt_mean_6h": last6["bt"].mean(),

            # Recent speed change
            "delta_speed": (
                last["speed"] - prev_speed
                if np.isfinite(prev_speed)
                else np.nan
            ),
        }

        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h IMF + solar wind features produced.")

    features = features.sort_values("timestamp").reset_index(drop=True)
    return features


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_8h_imf_solar_wind_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_8h(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h IMF + solar wind features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_8h_imf_solar_wind_features()


if __name__ == "__main__":
    main()
