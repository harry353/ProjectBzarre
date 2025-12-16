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
FLUX_COLS = [
    "irradiance_xrsa1",
    "irradiance_xrsa2",
    "irradiance_xrsb1",
    "irradiance_xrsb2",
    "xrs_ratio",
]

# GOES XRS quiet-Sun floor (W/m^2)
BACKGROUND_FLOOR = 1.0e-9


# ---------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------
def filter_xray_hourly() -> pd.DataFrame:
    df = read_timeseries_table(
        INPUT_TABLE,
        time_col="time_tag",
        value_cols=FLUX_COLS,
        db_path=INPUT_DB,
    )

    if df.empty:
        raise RuntimeError("Hourly X-ray dataset is empty.")

    # --------------------------------------------------------------
    # Missing-hour flag (all channels missing)
    # --------------------------------------------------------------
    df["xrs_missing_flag"] = (
        df[FLUX_COLS].isna().all(axis=1).astype(int)
    )

    # --------------------------------------------------------------
    # Background floor → 0
    # --------------------------------------------------------------
    for col in FLUX_COLS:
        df[col] = np.where(
            df[col] <= BACKGROUND_FLOOR,
            0.0,
            df[col],
        )

    # --------------------------------------------------------------
    # Clip negative values → 0
    # --------------------------------------------------------------
    df[FLUX_COLS] = np.maximum(df[FLUX_COLS], 0.0)

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
