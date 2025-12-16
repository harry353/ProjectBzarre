from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline.utils import write_sqlite_table  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parent
INPUT_DB = STAGE_DIR.parents[1] / "flares" / "1_combining" / "goes_flares_combined.db"
INPUT_TABLE = "goes_flares_combined"
OUTPUT_DB = STAGE_DIR / "flares_comb_filt.db"
OUTPUT_TABLE = "filtered_flares"
DROP_COLUMNS = {
    "flare_class",
    "flare_id",
    "satellite",
    "source_day",
    "file_url",
    "source_table",
}


def _load_combined() -> pd.DataFrame:
    if not INPUT_DB.exists():
        raise FileNotFoundError(f"Combined GOES flare database missing at {INPUT_DB}")

    with sqlite3.connect(INPUT_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {INPUT_TABLE}", conn)

    if "event_time" not in df.columns:
        raise RuntimeError(f"Table '{INPUT_TABLE}' missing 'event_time' column.")

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_time"])
    df = df.set_index("event_time").sort_index()
    return df


def apply_filters() -> pd.DataFrame:
    df = _load_combined()
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"] != "POST_EVENT"]
        removed_rows = before - len(df)
    else:
        removed_rows = 0

    filtered = df.drop(columns=DROP_COLUMNS.intersection(df.columns), errors="ignore")
    write_sqlite_table(filtered, OUTPUT_DB, OUTPUT_TABLE)
    print(
        f"[OK] Filtered GOES flare dataset saved to {OUTPUT_DB} "
        f"(removed columns: {', '.join(sorted(DROP_COLUMNS))}; "
        f"dropped {removed_rows} POST_EVENT rows)"
    )
    return filtered


def main() -> None:
    apply_filters()


if __name__ == "__main__":
    main()
