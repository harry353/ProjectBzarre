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

from preprocessing_pipeline.utils import read_timeseries_table, write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

INPUT_DB = (
    STAGE_DIR.parents[1]
    / "xray_flux"
    / "1_averaging"
    / "xray_flux_aver.db"
)
INPUT_TABLE = "hourly_data"

OUTPUT_DB = STAGE_DIR / "xray_flux_aver_filt.db"
OUTPUT_TABLE = "filtered_data"

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DETECTOR_COLS = {
    "xrsa": ["irradiance_xrsa1", "irradiance_xrsa2"],
    "xrsb": ["irradiance_xrsb1", "irradiance_xrsb2"],
}

FLUX_COLS = [
    *DETECTOR_COLS["xrsa"],
    *DETECTOR_COLS["xrsb"],
    "xrs_ratio",
]

# GOES XRS quiet-Sun floor (W/m^2)
BACKGROUND_FLOOR = 1.0e-9


# ---------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------
def filter_xray_hourly() -> pd.DataFrame:
    available_cols = FLUX_COLS.copy()
    try:
        df = read_timeseries_table(
            INPUT_TABLE,
            time_col="time_tag",
            value_cols=available_cols,
            db_path=INPUT_DB,
        )
    except pd.errors.DatabaseError:
        fallback_cols = ["irradiance_xrsa", "irradiance_xrsb", "xrs_ratio"]
        print("[WARN] Falling back to combined XRS columns for filtering:", fallback_cols)
        df = read_timeseries_table(
            INPUT_TABLE,
            time_col="time_tag",
            value_cols=fallback_cols,
            db_path=INPUT_DB,
        )
        available_cols = fallback_cols

    if df.empty:
        raise RuntimeError("Hourly X-ray dataset is empty.")

    # --------------------------------------------------------------
    # Missing-hour flag (all channels missing)
    # --------------------------------------------------------------
    df["xrs_missing_flag"] = (
        df[available_cols].isna().all(axis=1).astype(int)
    )

    # --------------------------------------------------------------
    # Background floor → 0
    # --------------------------------------------------------------
    for col in available_cols:
        df[col] = np.where(
            df[col] <= BACKGROUND_FLOOR,
            0.0,
            df[col],
        )

    # --------------------------------------------------------------
    # Clip negative values → 0
    # --------------------------------------------------------------
    df[available_cols] = np.maximum(df[available_cols], 0.0)

    # --------------------------------------------------------------
    # Median-combine redundant detectors (skip NaNs)
    # --------------------------------------------------------------
    missing_xrsa_detectors = any(col not in df.columns for col in DETECTOR_COLS["xrsa"])
    missing_xrsb_detectors = any(col not in df.columns for col in DETECTOR_COLS["xrsb"])

    if not missing_xrsa_detectors and not missing_xrsb_detectors:
        df["irradiance_xrsa"] = df[DETECTOR_COLS["xrsa"]].median(axis=1, skipna=True)
        df["irradiance_xrsb"] = df[DETECTOR_COLS["xrsb"]].median(axis=1, skipna=True)
        df = df.drop(columns=sum(DETECTOR_COLS.values(), []))
    else:
        # Already single channels present; ensure naming consistency
        if "irradiance_xrsa" not in df.columns and "irradiance_xrsa1" in df.columns:
            df["irradiance_xrsa"] = df["irradiance_xrsa1"]
        if "irradiance_xrsb" not in df.columns and "irradiance_xrsb1" in df.columns:
            df["irradiance_xrsb"] = df["irradiance_xrsb1"]

    write_sqlite_table(df, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] Filtered hourly X-ray flux written to {OUTPUT_DB}")
    print(f"     Total rows           : {len(df):,}")
    print(f"     Missing hours flagged: {df['xrs_missing_flag'].sum():,}")

    return df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    filter_xray_hourly()


if __name__ == "__main__":
    main()
