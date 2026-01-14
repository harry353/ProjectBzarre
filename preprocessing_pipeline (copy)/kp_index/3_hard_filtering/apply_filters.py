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
HOURLY_DB = STAGE_DIR.parents[1] / "kp_index" / "1_averaging" / "kp_index_aver.db"
HOURLY_TABLE = "hourly_data"
OUTPUT_DB = STAGE_DIR / "kp_index_aver_filt.db"
OUTPUT_TABLE = "filtered_data"
def apply_kp_filters() -> pd.DataFrame:
    df = load_hourly_output(HOURLY_DB, HOURLY_TABLE)
    if df.empty:
        raise RuntimeError("KP hourly data not found; run averaging step first.")
    write_sqlite_table(df, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] KP filtered dataset saved to {OUTPUT_DB}")
    return df


def main() -> None:
    apply_kp_filters()


if __name__ == "__main__":
    main()
