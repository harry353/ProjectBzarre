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
INPUT_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "2_concatenating_combining"
    / "imf_solar_wind_aver_comb.db"
)
INPUT_TABLE = "hourly_data"
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb_filt.db"
OUTPUT_TABLE = "filtered_data"
DATA_COLUMNS = ["density", "speed", "temperature", "bx_gse", "by_gse", "bz_gse", "bt"]
MISSING_THRESHOLD_HOURS = 12
MIN_TIMESTAMP = pd.Timestamp("1998-02-05T00:00:00Z")


def _drop_long_missing_runs(df: pd.DataFrame, threshold_hours: int) -> pd.DataFrame:
    data_cols = [col for col in DATA_COLUMNS if col in df.columns]
    if not data_cols:
        raise RuntimeError("No expected measurement columns found in input dataset.")

    missing_mask = df[data_cols].isna().all(axis=1)
    if not missing_mask.any():
        return df

    segments = (missing_mask != missing_mask.shift()).cumsum()
    run_lengths = missing_mask.groupby(segments).transform("sum")
    to_drop = missing_mask & (run_lengths > threshold_hours)
    return df.loc[~to_drop].copy()


def apply_filters() -> pd.DataFrame:
    df = load_hourly_output(INPUT_DB, INPUT_TABLE)
    if df.empty:
        raise RuntimeError("Combined IMF + solar wind data not found; run concatenation first.")

    filtered = df.loc[df.index >= MIN_TIMESTAMP].copy()
    filtered = _drop_long_missing_runs(filtered, MISSING_THRESHOLD_HOURS)
    dropped = len(df) - len(filtered)
    write_sqlite_table(filtered, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Filtered dataset saved to {OUTPUT_DB} (dropped {dropped} rows)")
    return filtered


def main() -> None:
    apply_filters()


if __name__ == "__main__":
    main()
