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

from preprocessing_pipeline.utils import read_timeseries_table, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
OUTPUT_DB = STAGE_DIR / "radio_flux_filt.db"
OUTPUT_TABLE = "filtered_data"
SOURCE_TABLE = "radio_flux"
TIME_COLUMN = "time_tag"
VALUE_COLUMNS = ["adjusted_flux"]


def apply_radio_flux_filters() -> pd.DataFrame:
    df = read_timeseries_table(
        SOURCE_TABLE,
        time_col=TIME_COLUMN,
        value_cols=VALUE_COLUMNS,
    )
    if df.empty:
        raise RuntimeError(f"No records found in table '{SOURCE_TABLE}'.")

    filtered = df.dropna(subset=VALUE_COLUMNS)
    write_sqlite_table(filtered, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Radio flux filtered dataset saved to {OUTPUT_DB}")
    return filtered


def main() -> None:
    apply_radio_flux_filters()


if __name__ == "__main__":
    main()
