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

from data_sources.cme.cme_download import download_cme_catalog

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DB = BASE_DIR / "cactus_cme_recent.db"
TABLE_NAME = "cactus_cme"


def fetch_recent_cme_events() -> Path:
    fetch_date = os.environ.get("FETCH_DATE")
    if fetch_date:
        target = date.fromisoformat(fetch_date)
        start = target - timedelta(days=7)
        end = target
        cutoff_start = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
        cutoff_end = cutoff_start + timedelta(days=1)
    else:
        end = date.today()
        start = end - timedelta(days=7)
        cutoff_end = datetime.now(timezone.utc)
        cutoff_start = cutoff_end - timedelta(hours=24)

    df = download_cme_catalog(start, end)
    if df.empty:
        filtered = pd.DataFrame(columns=df.columns)
    else:
        if df["time_tag"].dt.tz is None:
            df["time_tag"] = df["time_tag"].dt.tz_localize(timezone.utc)
        else:
            df["time_tag"] = df["time_tag"].dt.tz_convert(timezone.utc)
        filtered = df[(df["time_tag"] >= cutoff_start) & (df["time_tag"] < cutoff_end)]

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        filtered.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(
        f"[OK] Stored CACTUS CME events from last 24h ({len(filtered):,} rows) to {OUTPUT_DB}"
    )
    return OUTPUT_DB


if __name__ == "__main__":
    fetch_recent_cme_events()
