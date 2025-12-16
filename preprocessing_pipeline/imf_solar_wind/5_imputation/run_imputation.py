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

import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
FILTERED_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "4_hard_filtering"
    / "imf_solar_wind_aver_comb_filt.db"
)
FILTERED_TABLE = "filtered_data"
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb_filt_imp.db"
OUTPUT_TABLE = "imputed_data"

ESSENTIAL_COLUMNS = ["bx_gse", "by_gse", "bz_gse", "bt", "speed", "density", "temperature"]
SMALL_GAP_LIMIT = 3  # hours
LARGE_GAP_THRESHOLD = 13  # hours


def _load_filtered() -> pd.DataFrame:
    df = load_hourly_output(FILTERED_DB, FILTERED_TABLE)
    if df.empty:
        raise RuntimeError("Filtered IMF + solar wind dataset not found; run hard filtering first.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Filtered dataset must have a DatetimeIndex.")
    return df.sort_index()


def _impute_column(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    series = df[column].astype(float)
    imputed = series.interpolate(method="time", limit=SMALL_GAP_LIMIT, limit_direction="both")
    flags = pd.Series(0, index=df.index, dtype=int, name=f"{column}_missing_flag")

    orig_missing = series.isna()
    if not orig_missing.any():
        return imputed, flags

    groups = (orig_missing != orig_missing.shift()).cumsum()
    missing_groups = groups[orig_missing]

    for group_id in missing_groups.unique():
        run_mask = groups.eq(group_id)
        run_length = int(run_mask.sum())

        if run_length <= SMALL_GAP_LIMIT:
            if imputed.loc[run_mask].isna().any():
                imputed.loc[run_mask] = 0.0
                flags.loc[run_mask] = 1
            continue

        imputed.loc[run_mask] = 0.0
        flags.loc[run_mask] = 1

    imputed = imputed.fillna(0.0)
    return imputed, flags


def impute_imf_solar_wind() -> pd.DataFrame:
    df = _load_filtered()
    summary: dict[str, int] = {}

    for column in ESSENTIAL_COLUMNS:
        if column not in df.columns:
            raise RuntimeError(f"Required column '{column}' missing from filtered dataset.")
        imputed, flags = _impute_column(df, column)
        df[column] = imputed
        flag_col = f"{column}_missing_flag"
        df[flag_col] = flags
        summary[flag_col] = int(flags.sum())

    write_sqlite_table(df, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Imputed dataset saved to {OUTPUT_DB}")
    for column, count in summary.items():
        print(f"    - {column}: flagged {count} hours")
    return df


def main() -> None:
    impute_imf_solar_wind()


if __name__ == "__main__":
    main()
