from __future__ import annotations
import sys
from pathlib import Path

# Resolve absolute path of the current script and search for project root (space_weather_api.py)
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Inject project root into system path to allow local module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
# Source combined multi-instrument telemetry from the previous stage
INPUT_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "2_concatenating_combining"
    / "imf_solar_wind_aver_comb.db"
)
INPUT_TABLE = "hourly_data"
# Persist filtered results to a new staging database for this stage
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb_filt.db"
OUTPUT_TABLE = "filtered_data"
DATA_COLUMNS = ["density", "speed", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]
# Maximum allowed contiguous gap; segments exceeding this are dropped to avoid interpolation artifacts
MISSING_THRESHOLD_HOURS = 12
# Global mission-start cutoff for uniform data availability
MIN_TIMESTAMP = pd.Timestamp("1998-02-05T00:00:00Z")


# Identifies and removes contiguous blocks of missing data exceeding a specific duration
def _drop_long_missing_runs(df: pd.DataFrame, threshold_hours: int) -> pd.DataFrame:
    data_cols = [col for col in DATA_COLUMNS if col in df.columns]
    if not data_cols:
        raise RuntimeError("No expected measurement columns found in input dataset.")

    # A 'run' is defined as consecutive rows where all telemetry parameters are NaN
    missing_mask = df[data_cols].isna().all(axis=1)
    if not missing_mask.any():
        return df

    # Vectorized run identification using cumulative sum of mask transitions
    segments = (missing_mask != missing_mask.shift()).cumsum()
    run_lengths = missing_mask.groupby(segments).transform("sum")
    
    # Isolate runs that exceed the allowed interpolation limit
    to_drop = missing_mask & (run_lengths > threshold_hours)
    return df.loc[~to_drop].copy()


# Orchestrates data cleaning: temporal clipping, gap pruning, and physical validity checks
def apply_filters() -> pd.DataFrame:
    df = load_hourly_output(INPUT_DB, INPUT_TABLE)
    if df.empty:
        raise RuntimeError("Combined IMF + solar wind data not found; run concatenation first.")

    # 1. Enforce a global start-time cutoff
    filtered = df.loc[df.index >= MIN_TIMESTAMP].copy()
    
    # 2. Prune data segments with excessive missingness
    filtered = _drop_long_missing_runs(filtered, MISSING_THRESHOLD_HOURS)
    
    # 3. Enforce physical constraints: Solar Wind parameters must be non-negative
    non_negative_cols = [col for col in ("density", "speed", "temperature") if col in filtered.columns]
    if not non_negative_cols:
        raise RuntimeError("No expected solar wind columns found in input dataset.")
    
    # Drop rows with non-physical (e.g., negative) telemetry values
    filtered = filtered.loc[~(filtered[non_negative_cols] < 0).any(axis=1)].copy()
    dropped = len(df) - len(filtered)
    
    # Persist the cleaned dataset
    write_sqlite_table(filtered, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Filtered dataset saved to {OUTPUT_DB} (dropped {dropped} rows)")
    return filtered


def main() -> None:
    apply_filters()


if __name__ == "__main__":
    main()
