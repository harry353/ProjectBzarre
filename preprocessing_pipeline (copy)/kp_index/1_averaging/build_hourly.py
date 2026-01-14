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

from preprocessing_pipeline.utils import read_timeseries_table, resample_to_hourly, write_sqlite_table

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DB = BASE_DIR / "kp_index_aver.db"
OUTPUT_TABLE = "hourly_data"
SOURCE_TABLE = "kp_index"
TIME_COLUMN = "time_tag"
VALUE_COLUMNS = ["kp_index"]
RESAMPLE_METHOD = "ffill"


def build_kp_hourly() -> pd.DataFrame:
    df = read_timeseries_table(
        SOURCE_TABLE,
        time_col=TIME_COLUMN,
        value_cols=VALUE_COLUMNS,
    )
    if df.empty:
        raise RuntimeError(f"No records found in table '{SOURCE_TABLE}'.")
    hourly = resample_to_hourly(df, method=RESAMPLE_METHOD)
    write_sqlite_table(hourly, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] KP hourly dataset written to {OUTPUT_DB}")
    return hourly


def main() -> None:
    build_kp_hourly()


if __name__ == "__main__":
    main()
