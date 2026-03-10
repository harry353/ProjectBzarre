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

from preprocessing_pipeline.utils import read_timeseries_table, resample_to_hourly, write_sqlite_table

BASE_DIR = Path(__file__).resolve().parent
# Final database for the hourly averaged Kp-Index
OUTPUT_DB = BASE_DIR / "kp_index_aver.db"
OUTPUT_TABLE = "hourly_data"
SOURCE_TABLE = "kp_index"
TIME_COLUMN = "time_tag"
VALUE_COLUMNS = ["kp_index"]
# Use forward fill for Kp-Index since it represents a 3-hour interval (constant until next report)
RESAMPLE_METHOD = "ffill"


# Loads raw KP data, aggregates to hourly bins using forward fill, and persists result
def build_kp_hourly() -> pd.DataFrame:
    # Query raw indices from the telemetry database
    df = read_timeseries_table(
        SOURCE_TABLE,
        time_col=TIME_COLUMN,
        value_cols=VALUE_COLUMNS,
    )
    if df.empty:
        raise RuntimeError(f"No records found in table '{SOURCE_TABLE}'.")
    
    # Perform temporal resampling to align with the uniform hourly pipeline grid
    hourly = resample_to_hourly(df, method=RESAMPLE_METHOD)
    
    # Save processed hourly data to a local SQLite database for downstream stages
    write_sqlite_table(hourly, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] KP hourly dataset written to {OUTPUT_DB}")
    return hourly


def main() -> None:
    build_kp_hourly()


if __name__ == "__main__":
    main()
