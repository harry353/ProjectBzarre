from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DB = BASE_DIR / "dst_recent.db"
TABLE_NAME = "dst_recent"
SOURCE_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"


def _download_json() -> list:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def _normalize_payload(payload: list) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    if isinstance(payload[0], dict):
        df = pd.DataFrame(payload)
    elif isinstance(payload[0], list):
        header = payload[0]
        rows = payload[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        return pd.DataFrame()

    lower = {col.lower(): col for col in df.columns}
    time_col = lower.get("time_tag") or lower.get("time") or lower.get("date")
    dst_col = lower.get("dst")
    if time_col is None or dst_col is None:
        raise RuntimeError("Unexpected Kyoto DST JSON format.")

    df = df.rename(columns={time_col: "time_tag", dst_col: "dst"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    return df.dropna(subset=["time_tag"])


def fetch_recent_dst() -> Path:
    fetch_date = os.environ.get("FETCH_DATE")
    if fetch_date:
        target = date.fromisoformat(fetch_date)
        start = target
        end = target
        cutoff_start = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
        cutoff_end = cutoff_start + timedelta(days=1)
    else:
        today = date.today()
        start = today - timedelta(days=2)
        end = today
        cutoff_end = datetime.now(timezone.utc)
        cutoff_start = cutoff_end - timedelta(hours=24)

    payload = _download_json()
    df = _normalize_payload(payload)

    if not df.empty:
        df["time_tag"] = df["time_tag"].dt.tz_convert(timezone.utc)
        cutoff_start = cutoff_start.replace(minute=0, second=0, microsecond=0)
        cutoff_end = cutoff_end.replace(minute=0, second=0, microsecond=0)
        df = df[(df["time_tag"] >= cutoff_start) & (df["time_tag"] < cutoff_end)]
        df = df.dropna(subset=["dst"]).sort_values("time_tag").tail(24)

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"[OK] Stored {len(df):,} DST rows from last 24h to {OUTPUT_DB}")
    return OUTPUT_DB


if __name__ == "__main__":
    fetch_recent_dst()
