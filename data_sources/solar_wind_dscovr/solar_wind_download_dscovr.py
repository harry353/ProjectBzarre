from __future__ import annotations

import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import List, Optional

import pandas as pd
import xarray as xr
import requests
from common.http import http_get
from space_weather_api import format_date
from database_builder.logging_utils import stamp

from database_builder.constants import BUILD_FROM_REALTIME, REALTIME_BACKFILL_DAYS

BASE_DIR = "https://www.ngdc.noaa.gov/dscovr/data"
COLUMNS = ["time_tag", "density", "speed", "temperature", "source_type"]
SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"


def download_solar_wind_dscovr(start_date, end_date, max_workers: int = 8) -> pd.DataFrame:
    """
    Download DSCOVR F1M plasma NetCDF files and combine them.
    """
    days: List = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)

    if not days:
        return pd.DataFrame(columns=COLUMNS)

    results: List[pd.DataFrame] = []
    missing_days: List = []
    missing_lock = threading.Lock()

    def process_day(day):
        day_df, missing = _download_day(day)
        if missing:
            with missing_lock:
                missing_days.append(day)
        if day_df is not None and not day_df.empty:
            results.append(day_df)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_day, day): day for day in days}
        for future in as_completed(future_map):
            try:
                future.result()
            except Exception as exc:
                day = future_map[future]
                print(f"[ERROR] DSCOVR day {format_date(day)} failed: {exc}")

    _emit_missing_ranges("DSCOVR F1M", missing_days)

    if not results:
        return pd.DataFrame(columns=COLUMNS)

    combined = pd.concat(results).sort_values("time_tag").reset_index(drop=True)
    combined["source_type"] = "archive"

    if BUILD_FROM_REALTIME and REALTIME_BACKFILL_DAYS > 0 and end_date >= start_date:
        realtime_start = end_date - timedelta(days=REALTIME_BACKFILL_DAYS - 1)
        start_dt = datetime.combine(realtime_start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        realtime_df = _download_swpc_plasma(start_dt, end_dt)
        if not realtime_df.empty:
            combined = pd.concat([combined, realtime_df], ignore_index=True)
            combined["time_tag"] = pd.to_datetime(combined["time_tag"], errors="coerce", utc=True)
            combined = combined.dropna(subset=["time_tag"]).sort_values("time_tag")
            combined = combined.drop_duplicates(subset="time_tag", keep="last").reset_index(drop=True)

    return combined


def _download_day(day) -> tuple[Optional[pd.DataFrame], bool]:
    directory = f"{BASE_DIR}/{day.year}/{day.month:02d}/"

    response = http_get(directory, log_name="Solar Wind DSCOVR", timeout=15)
    if response is None:
        return None, True
    html = response.text

    day_str = day.strftime("%Y%m%d000000")
    pattern = rf"(oe_f1m_dscovr_s{day_str}_e\d+_p\d+_pub\.nc\.gz)"
    matches = re.findall(pattern, html)

    if not matches:
        return None, True

    filename = matches[0]
    file_url = directory + filename

    response = http_get(file_url, log_name="Solar Wind DSCOVR", timeout=20)
    if response is None:
        return None, True
    gz_bytes = response.content

    try:
        with gzip.open(BytesIO(gz_bytes), "rb") as fh:
            nc_bytes = fh.read()
    except Exception as exc:
        print(f"[WARN] Could not decompress {filename}: {exc}")
        return None, False

    try:
        ds = xr.open_dataset(BytesIO(nc_bytes), engine="scipy")
    except Exception as exc:
        print(f"[WARN] Failed to open NC file {filename}: {exc}")
        return None, False

    df = ds.to_dataframe().reset_index()
    ds.close()
    return _extract(df), False


def _emit_missing_ranges(label: str, days: list) -> None:
    if not days:
        return
    days = sorted(set(days))
    ranges = []
    start = prev = days[0]
    for day in days[1:]:
        if (day - prev).days == 1:
            prev = day
            continue
        ranges.append((start, prev))
        start = prev = day
    ranges.append((start, prev))
    for start, end in ranges:
        if start == end:
            print(stamp(f"[WARN] No {label} match for {format_date(start)}"))
        else:
            print(
                stamp(
                    f"[WARN] No {label} match for {format_date(start)} -> {format_date(end)}"
                )
            )


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    time_column = None
    for candidate in ("time", "Epoch", "epoch", "TIME_TAG", "time_tag"):
        if candidate in df.columns:
            time_column = candidate
            break

    if time_column is None:
        # Ensure we keep a placeholder column so downstream logic can handle it.
        df["time"] = None
        time_column = "time"

    rename_map = {time_column: "time_tag"}
    for source, target in [
        ("proton_density", "density"),
        ("proton_speed", "speed"),
        ("proton_temperature", "temperature"),
    ]:
        if source in df.columns:
            rename_map[source] = target
        else:
            df[target] = None

    subset_cols = ["time_tag", "density", "speed", "temperature"]
    df = df.rename(columns=rename_map)
    df = df.reindex(columns=subset_cols, fill_value=None).copy()
    df["time_tag"] = _coerce_time(df["time_tag"])
    return df


def _coerce_time(series: pd.Series) -> pd.Series:
    """
    Convert the DSCOVR time column into UTC timestamps even when the NetCDF
    file leaves the values as raw seconds since epoch or string tokens.
    """
    # 1) Direct datetime parsing (handles ISO strings and datetime64 columns)
    direct = pd.to_datetime(series, errors="coerce", utc=True)
    if direct.notna().any():
        return direct

    # 2) Try numeric epochs in s/ms/us/ns
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        for unit in ("s", "ms", "us", "ns"):
            converted = pd.to_datetime(
                numeric, errors="coerce", utc=True, unit=unit, origin="unix"
            )
            if converted.notna().any():
                return converted

    # 3) Fall back to naive parsing (may still yield NaT if input invalid)
    return pd.to_datetime(series.astype(str), errors="coerce", utc=True)


def _download_swpc_plasma(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        resp = requests.get(SWPC_PLASMA_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=COLUMNS)

    if isinstance(payload[0], list):
        header = payload[0]
        rows = payload[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.DataFrame(payload)

    lower = {col.lower(): col for col in df.columns}
    time_col = lower.get("time_tag") or lower.get("time") or lower.get("timestamp")
    if time_col is None:
        return pd.DataFrame(columns=COLUMNS)

    rename = {time_col: "time_tag"}
    for src, dst in [("density", "density"), ("speed", "speed"), ("temperature", "temperature")]:
        if src in lower:
            rename[lower[src]] = dst
        else:
            df[dst] = None

    df = df.rename(columns=rename)
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df["density"] = pd.to_numeric(df["density"], errors="coerce")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.dropna(subset=["time_tag"])
    df = df[(df["time_tag"] >= start_dt) & (df["time_tag"] <= end_dt)]
    df["source_type"] = "realtime"
    return df[COLUMNS].reset_index(drop=True)
