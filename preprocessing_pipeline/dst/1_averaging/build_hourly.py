from __future__ import annotations
import sys
from pathlib import Path

# Resolve the absolute path of the current script
PROJECT_ROOT = Path(__file__).resolve()
# Traverse up the directory tree to find the project root (marked by space_weather_api.py)
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Ensure the project root is in the system path for local module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from preprocessing_pipeline.utils import read_timeseries_table, resample_to_hourly, write_sqlite_table

BASE_DIR = Path(__file__).resolve().parent
# Staging database for averaged DST indices
OUTPUT_DB = BASE_DIR / "dst_aver.db"
OUTPUT_TABLE = "hourly_data"
SOURCE_TABLE = "dst_index"
TIME_COLUMN = "time_tag"
VALUE_COLUMNS = ["dst"]
# Use forward-fill to maintain continuity during hourly resampling
RESAMPLE_METHOD = "ffill"


# Loads DST index data, resamples it to a uniform hourly grid, and persists the result
def build_hourly_dst() -> pd.DataFrame:
    # Retrieve raw time-series from the central database
    df = read_timeseries_table(
        SOURCE_TABLE,
        time_col=TIME_COLUMN,
        value_cols=VALUE_COLUMNS,
    )
    if df.empty:
        raise RuntimeError(f"No records found in table '{SOURCE_TABLE}'.")
    # Align data to strictly hourly timestamps
    hourly = resample_to_hourly(df, method=RESAMPLE_METHOD)
    # Save the consolidated hourly dataset to the local staging DB
    write_sqlite_table(hourly, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] DST hourly dataset written to {OUTPUT_DB}")
    return hourly


def main() -> None:
    build_hourly_dst()


if __name__ == "__main__":
    main()
