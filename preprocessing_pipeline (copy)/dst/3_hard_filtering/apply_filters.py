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
HOURLY_DB = STAGE_DIR.parents[1] / "dst" / "1_averaging" / "dst_aver.db"
HOURLY_TABLE = "hourly_data"
OUTPUT_DB = STAGE_DIR / "dst_aver_filt.db"
OUTPUT_TABLE = "filtered_data"
REQUIRED_COLUMNS = ["dst"]


def apply_dst_filters() -> pd.DataFrame:
    df = load_hourly_output(HOURLY_DB, HOURLY_TABLE)
    if df.empty:
        raise RuntimeError("DST hourly data not found; run averaging step first.")
    filtered = df.dropna(subset=REQUIRED_COLUMNS)
    write_sqlite_table(filtered, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] DST filtered dataset saved to {OUTPUT_DB}")
    return filtered


def main() -> None:
    apply_dst_filters()


if __name__ == "__main__":
    main()
