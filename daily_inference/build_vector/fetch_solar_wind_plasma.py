from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, List

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DB = BASE_DIR / "solar_wind_plasma_1day.db"
TABLE_NAME = "solar_wind_plasma"
SOURCE_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"

PREFERRED_COLS = [
    "time_tag",
    "density",
    "speed",
    "temperature",
]


def _download_json(url: str) -> List[Any]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload from {url}: {type(payload)}")
    return payload


def _normalize_dataframe(raw: List[Any]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    if isinstance(raw[0], list):
        header = raw[0]
        rows = raw[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.DataFrame(raw)
    if df.empty:
        return df

    time_col = None
    for candidate in ("time_tag", "time", "timestamp"):
        if candidate in df.columns:
            time_col = candidate
            break
    if not time_col:
        raise ValueError("Solar wind plasma JSON missing timestamp column.")

    df["time_tag"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["time_tag"])

    for col in df.columns:
        if col == "time_tag" or col == time_col:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ordered = [c for c in PREFERRED_COLS if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered and c != time_col]
    df = df[ordered + remaining].sort_values("time_tag")
    fetch_date = os.environ.get("FETCH_DATE")
    if fetch_date:
        target = pd.Timestamp(fetch_date, tz="UTC")
        end = target + pd.Timedelta(days=1)
        df = df[(df["time_tag"] >= target) & (df["time_tag"] < end)]
    else:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
        df = df[df["time_tag"] >= cutoff]
    return df


def fetch_and_store() -> Path:
    payload = _download_json(SOURCE_URL)
    df = _normalize_dataframe(payload)

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"[OK] Stored solar wind plasma JSON ({len(df):,} rows) to {OUTPUT_DB}")
    return OUTPUT_DB


if __name__ == "__main__":
    fetch_and_store()
