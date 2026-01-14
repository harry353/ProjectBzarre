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

# ------------------------------------------------------------------
# Aggregation parameters (authoritative)
# ------------------------------------------------------------------

MIN_FRACTION_COVERAGE = 0.5

BZ_MIN_WINDOW_H = 6
BZ_SOUTH_FRAC_WINDOW_H = 6
NEWELL_INT_WINDOW_H = 6
EY_MEAN_WINDOW_H = 6
PDYN_MAX_WINDOW_H = 6


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
# Build aggregates (PAST-ONLY)
# ---------------------------------------------------------------------
def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    out = df.copy()

    def _min_periods(w: int) -> int:
        return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))

    # --------------------------------------------------------------
    # Strong southward IMF
    # --------------------------------------------------------------
    w = BZ_MIN_WINDOW_H
    window = f"{w}h"
    out[f"bz_min_{w}h"] = (
        df["bz_gse"]
        .rolling(window, min_periods=_min_periods(w))
        .min()
    )

    # --------------------------------------------------------------
    # Southward IMF persistence
    # --------------------------------------------------------------
    w = BZ_SOUTH_FRAC_WINDOW_H
    window = f"{w}h"
    out[f"bz_south_frac_{w}h"] = (
        (df["bz_gse"] < 0)
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    # --------------------------------------------------------------
    # Integrated coupling (Newell)
    # --------------------------------------------------------------
    w = NEWELL_INT_WINDOW_H
    window = f"{w}h"
    out[f"newell_int_{w}h"] = (
        df["newell_dphi_dt"]
        .rolling(window, min_periods=_min_periods(w))
        .sum()
    )

    # --------------------------------------------------------------
    # Mean reconnection electric field
    # --------------------------------------------------------------
    w = EY_MEAN_WINDOW_H
    window = f"{w}h"
    out[f"ey_mean_{w}h"] = (
        df["ey"]
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    # --------------------------------------------------------------
    # Sustained compression
    # --------------------------------------------------------------
    w = PDYN_MAX_WINDOW_H
    window = f"{w}h"
    out[f"pdyn_max_{w}h"] = (
        df["dynamic_pressure"]
        .rolling(window, min_periods=_min_periods(w))
        .max()
    )

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------
    out = out.dropna()

    if out.empty:
        raise RuntimeError("No aggregated IMF + solar wind features produced.")

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_8h_imf_solar_wind_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_agg(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h IMF + solar wind features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_8h_imf_solar_wind_features()


if __name__ == "__main__":
    main()
