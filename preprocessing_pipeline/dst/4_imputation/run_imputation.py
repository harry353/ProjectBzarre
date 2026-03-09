from __future__ import annotations
import sys
from pathlib import Path

# Handle absolute path resolution with fallbacks for local and installed modes
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
# Preferred source: results from the filtering stage
FILTERED_DB = STAGE_DIR.parents[1] / "dst" / "3_hard_filtering" / "dst_aver_filt.db"
FILTERED_TABLE = "filtered_data"
# Fallback source: raw averaged data if filtering was skipped
HOURLY_DB = STAGE_DIR.parents[1] / "dst" / "1_averaging" / "dst_aver.db"
HOURLY_TABLE = "hourly_data"
OUTPUT_DB = STAGE_DIR / "dst_aver_filt_imp.db"
OUTPUT_TABLE = "imputed_data"


# Attempts to load filtered data; falls back to raw averaged data if necessary
def _load_filtered() -> pd.DataFrame:
    try:
        return load_hourly_output(FILTERED_DB, FILTERED_TABLE)
    except FileNotFoundError:
        return load_hourly_output(HOURLY_DB, HOURLY_TABLE)


# Fills gaps in the DST time-series using a combination of interpolation and padding
def impute_dst() -> pd.DataFrame:
    df = _load_filtered()
    if df.empty:
        raise RuntimeError("No DST data available for imputation.")

    working = df.copy()
    missing_mask = working["dst"].isna()
    
    # 1. Time-weighted interpolation for short gaps (max 3 hours)
    working["dst"] = working["dst"].interpolate(method="time", limit=3, limit_direction="both")
    # 2. Forward and backward filling for remaining contiguous gaps
    working["dst"] = working["dst"].ffill().bfill()
    # 3. Final safety fill for edge cases where no data exists at all
    working["dst"] = working["dst"].fillna(0.0)

    # Persist the fully imputed dataset
    write_sqlite_table(working, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Imputed DST dataset saved to {OUTPUT_DB}")
    return working


def main() -> None:
    impute_dst()


if __name__ == "__main__":
    main()
