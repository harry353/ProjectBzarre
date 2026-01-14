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
    / "3_hard_filtering"
    / "xray_flux_aver_filt.db"
)
INPUT_TABLE = "filtered_data"

OUTPUT_DB = STAGE_DIR / "xray_flux_aver_filt_imp.db"
OUTPUT_TABLE = "imputed_data"

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
FLUX_COLS = [
    "irradiance_xrsa",
    "irradiance_xrsb",
    "xrs_ratio",
]

# ---------------------------------------------------------------------
# Imputation logic
# ---------------------------------------------------------------------
def impute_xray_hourly() -> pd.DataFrame:
    df = read_timeseries_table(
        INPUT_TABLE,
        time_col="time_tag",
        value_cols=FLUX_COLS + ["xrs_missing_flag"],
        db_path=INPUT_DB,
    )

    if df.empty:
        raise RuntimeError("Filtered X-ray dataset is empty.")

    # --------------------------------------------------------------
    # Imputation: NaN → 0 (explicit pipeline stage)
    # --------------------------------------------------------------
    df[FLUX_COLS] = df[FLUX_COLS].fillna(0.0)

    # --------------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------------
    if df[FLUX_COLS].isna().any().any():
        raise RuntimeError("NaNs remain after XRS imputation stage.")

    write_sqlite_table(df, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] Imputed hourly X-ray flux written to {OUTPUT_DB}")
    print(f"     Total rows: {len(df):,}")
    print(f"     Missing hours flagged: {df['xrs_missing_flag'].sum():,}")

    return df


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    impute_xray_hourly()


if __name__ == "__main__":
    main()
