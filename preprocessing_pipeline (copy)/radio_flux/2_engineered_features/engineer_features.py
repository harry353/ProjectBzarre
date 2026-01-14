from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

FILTERED_DB = (
    STAGE_DIR.parents[1]
    / "radio_flux"
    / "1_hard_filtering"
    / "radio_flux_filt.db"
)
FILTERED_TABLE = "filtered_data"

OUTPUT_DB = STAGE_DIR / "radio_flux_filt_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPS = 1e-6
SOLAR_ROTATION_HOURS = 27 * 24

# ---------------------------------------------------------------------
# Feature engineering (MINIMAL, HOURLY, CAUSAL)
# ---------------------------------------------------------------------
def _add_radio_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "adjusted_flux" not in working.columns:
        raise RuntimeError("Expected column 'adjusted_flux' missing.")

    # -----------------------------------------------------------------
    # Rename + ensure time index
    # -----------------------------------------------------------------
    working = working.rename(columns={"adjusted_flux": "f107"})

    if not isinstance(working.index, pd.DatetimeIndex):
        raise RuntimeError("Radio flux data must be time-indexed.")

    working = working.sort_index()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    else:
        working.index = working.index.tz_convert("UTC")

    # -----------------------------------------------------------------
    # Resample to hourly cadence (forward-fill, causal)
    # -----------------------------------------------------------------
    hourly_index = pd.date_range(
        working.index.min().floor("h"),
        working.index.max().ceil("h"),
        freq="1h",
        tz="UTC",
    )

    working = (
        working
        .reindex(hourly_index)
        .ffill()
    )

    # -----------------------------------------------------------------
    # Engineered features (ONLY 4)
    # -----------------------------------------------------------------
    working["log_f107"] = np.log(working["f107"].clip(lower=EPS))

    # Day-scale trend
    working["df107_24h"] = working["f107"] - working["f107"].shift(24)

    # Solar-rotation anomaly
    rolling_mean_27d = (
        working["f107"]
        .rolling(SOLAR_ROTATION_HOURS, min_periods=24)
        .mean()
    )
    working["f107_anomaly_27d"] = working["f107"] - rolling_mean_27d

    engineered = [
        "f107",
        "log_f107",
        "df107_24h",
        "f107_anomaly_27d",
    ]

    working[engineered] = working[engineered].fillna(0.0)

    if working[engineered].isna().any().any():
        raise RuntimeError("NaNs detected after radio flux feature engineering.")

    return working[engineered]


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def engineer_radio_flux_features() -> pd.DataFrame:
    df = load_hourly_output(FILTERED_DB, FILTERED_TABLE)
    if df.empty:
        raise RuntimeError("Filtered radio flux dataset not found.")

    features = _add_radio_flux_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] Radio flux engineered features written to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_radio_flux_features()


if __name__ == "__main__":
    main()
