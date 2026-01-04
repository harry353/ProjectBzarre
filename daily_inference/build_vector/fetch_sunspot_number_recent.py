from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys
import os

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

from data_sources.sunspot_number.sunspot_number_download import download_sunspot_numbers

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DB = BASE_DIR / "sunspot_number_recent.db"
TABLE_NAME = "sunspot_number_recent"


def fetch_recent_sunspot_number() -> Path:
    fetch_date = os.environ.get("FETCH_DATE")
    if fetch_date:
        target = date.fromisoformat(fetch_date)
        start = target
        end = target
        cutoff_start = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
        cutoff_end = cutoff_start + timedelta(days=1)
    else:
        end = date.today()
        start = end - timedelta(days=7)
        cutoff_start = None
        cutoff_end = None

    df = download_sunspot_numbers(start, end)

    if not df.empty:
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df = df.sort_values("time_tag")
        if cutoff_start is not None and cutoff_end is not None:
            df = df[(df["time_tag"] >= cutoff_start) & (df["time_tag"] < cutoff_end)]
        else:
            df = df.tail(1)

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"[OK] Stored {len(df):,} sunspot rows from last 24h to {OUTPUT_DB}")
    return OUTPUT_DB


if __name__ == "__main__":
    fetch_recent_sunspot_number()
