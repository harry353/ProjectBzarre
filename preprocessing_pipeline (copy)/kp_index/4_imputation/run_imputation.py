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
FILTERED_DB = STAGE_DIR.parents[1] / "kp_index" / "3_hard_filtering" / "kp_index_aver_filt.db"
FILTERED_TABLE = "filtered_data"
HOURLY_DB = STAGE_DIR.parents[1] / "kp_index" / "1_averaging" / "kp_index_aver.db"
HOURLY_TABLE = "hourly_data"
OUTPUT_DB = STAGE_DIR / "kp_index_aver_filt_imp.db"
OUTPUT_TABLE = "imputed_data"


def _load_filtered() -> pd.DataFrame:
    try:
        return load_hourly_output(FILTERED_DB, FILTERED_TABLE)
    except FileNotFoundError:
        return load_hourly_output(HOURLY_DB, HOURLY_TABLE)


def impute_kp() -> pd.DataFrame:
    df = _load_filtered()
    if df.empty:
        raise RuntimeError("No KP data available for imputation.")

    write_sqlite_table(df, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] KP dataset copied to {OUTPUT_DB}")
    return df


def main() -> None:
    impute_kp()


if __name__ == "__main__":
    main()
