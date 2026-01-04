from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import requests
import shutil
import sqlite3


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/haris/Documents/ProjectBzarre")
OUTPUT_DB = BASE_DIR / "goes_xrays_1day.db"
TABLE_NAME = "xray_flux"
SOURCE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


def _download_json(url: str) -> List[dict[str, Any]]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected payload from {url}: {type(payload)}")
    return payload


def _normalize_dataframe(raw: List[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    time_col = None
    for candidate in ("time_tag", "time", "timestamp"):
        if candidate in df.columns:
            time_col = candidate
            break
    if not time_col:
        raise ValueError("GOES X-ray JSON missing timestamp column.")

    df["time_tag"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    if "flux" in df.columns:
        value_col = "flux"
    elif "irradiance" in df.columns:
        value_col = "irradiance"
    else:
        raise ValueError("GOES X-ray JSON missing flux/irradiance column.")

    pivot = (
        df.pivot_table(
            index="time_tag",
            columns="energy",
            values=value_col,
            aggfunc="median",
        )
        .reset_index()
        .dropna(subset=["time_tag"])
    )

    pivot["irradiance_xrsb"] = pd.to_numeric(
        pivot.get("0.1-0.8nm"), errors="coerce"
    )
    pivot["irradiance_xrsa"] = pd.to_numeric(
        pivot.get("0.05-0.4nm"), errors="coerce"
    )

    pivot = pivot.dropna(
        subset=["irradiance_xrsa", "irradiance_xrsb"], how="all"
    )
    pivot["xrs_ratio"] = pivot["irradiance_xrsa"] / pivot["irradiance_xrsb"].replace(0, np.nan)
    ordered_cols = ["time_tag", "irradiance_xrsa", "irradiance_xrsb", "xrs_ratio"]
    pivot = pivot[ordered_cols].sort_values("time_tag")
    fetch_date = os.environ.get("FETCH_DATE")
    if fetch_date:
        target = pd.Timestamp(fetch_date, tz="UTC")
        end = target + pd.Timedelta(days=1)
        pivot = pivot[(pivot["time_tag"] >= target) & (pivot["time_tag"] < end)]
    else:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
        pivot = pivot[pivot["time_tag"] >= cutoff]
    return pivot


def fetch_and_store() -> Path:
    payload = _download_json(SOURCE_URL)
    df = _normalize_dataframe(payload)

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"[OK] Stored GOES X-ray JSON ({len(df):,} rows) to {OUTPUT_DB}")

    return OUTPUT_DB


if __name__ == "__main__":
    fetch_and_store()
