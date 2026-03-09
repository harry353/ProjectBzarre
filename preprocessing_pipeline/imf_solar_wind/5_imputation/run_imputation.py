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

# Inject project root into system path to enable module imports from the local codebase
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
# Source filtered IMF + solar wind data from the previous stage
FILTERED_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "4_hard_filtering"
    / "imf_solar_wind_aver_comb_filt.db"
)
FILTERED_TABLE = "filtered_data"
# Persist imputed results to a new staging database
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb_filt_imp.db"
OUTPUT_TABLE = "imputed_data"

# Core physical parameters required for downstream modeling
ESSENTIAL_COLUMNS = ["bx_gse", "by_gse", "bz_gse", "bt", "speed", "density", "temperature"]
# Threshold for linear interpolation; gaps larger than this are flagged as missing
SMALL_GAP_LIMIT = 3  # hours
LARGE_GAP_THRESHOLD = 13  # hours


# Loads filtered telemetry and ensures chronological ordering
def _load_filtered() -> pd.DataFrame:
    df = load_hourly_output(FILTERED_DB, FILTERED_TABLE)
    if df.empty:
        raise RuntimeError("Filtered IMF + solar wind dataset not found; run hard filtering first.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Filtered dataset must have a DatetimeIndex.")
    return df.sort_index()


# Implements a hybrid imputation: time-interpolation for short gaps, zero-fill + flag for long gaps
def _impute_column(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    series = df[column].astype(float)
    # Attempt time-weighted interpolation for all gaps (constrained by limit later)
    imputed = series.interpolate(method="time", limit=SMALL_GAP_LIMIT, limit_direction="both")
    # Initialize a binary flag column (1 = data was missing/imputed, 0 = raw observation)
    flags = pd.Series(0, index=df.index, dtype=int, name=f"{column}_missing_flag")

    orig_missing = series.isna()
    if not orig_missing.any():
        return imputed, flags

    # Group contiguous missing blocks to measure their length
    groups = (orig_missing != orig_missing.shift()).cumsum()
    missing_groups = groups[orig_missing]

    for group_id in missing_groups.unique():
        run_mask = groups.eq(group_id)
        run_length = int(run_mask.sum())

        # For gaps <= 3 hours, we accept the interpolation
        if run_length <= SMALL_GAP_LIMIT:
            # If interpolation failed (e.g., at edges), fallback to zero and flag
            if imputed.loc[run_mask].isna().any():
                imputed.loc[run_mask] = 0.0
                flags.loc[run_mask] = 1
            continue

        # For gaps > 3 hours, the signal is considered unreliable; zero-fill and flag
        imputed.loc[run_mask] = 0.0
        flags.loc[run_mask] = 1

    # Ensure no NaNs remain in the telemetry stream
    imputed = imputed.fillna(0.0)
    return imputed, flags


# Orchestrates the imputation process across all essential telemetry channels
def impute_imf_solar_wind() -> pd.DataFrame:
    df = _load_filtered()
    summary: dict[str, int] = {}

    for column in ESSENTIAL_COLUMNS:
        if column not in df.columns:
            raise RuntimeError(f"Required column '{column}' missing from filtered dataset.")
        
        # Apply the hybrid imputation strategy per column
        imputed, flags = _impute_column(df, column)
        df[column] = imputed
        flag_col = f"{column}_missing_flag"
        df[flag_col] = flags
        summary[flag_col] = int(flags.sum())

    # Save the fully imputed and flagged dataset
    write_sqlite_table(df, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Imputed dataset saved to {OUTPUT_DB}")
    for column, count in summary.items():
        print(f"    - {column}: flagged {count} hours")
    return df


def main() -> None:
    impute_imf_solar_wind()


if __name__ == "__main__":
    main()
