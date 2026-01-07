import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pandas as pd
import requests
import xarray as xr

from common.http import http_get

from space_weather_api import format_date
from database_builder.logging_utils import stamp
from database_builder.constants import BUILD_FROM_REALTIME, REALTIME_BACKFILL_DAYS

BASE_DIR = "https://www.ngdc.noaa.gov/dscovr/data"
M1M_COLUMNS = ["time_tag", "bt", "bx", "by", "bz", "source_type"]
SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"


def download_imf_discovr(start_date, end_date, max_workers=8):
    """
    Download and concatenate DSCOVR M1M files for the date range.
    """
    days = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)

    if not days:
        return pd.DataFrame(columns=M1M_COLUMNS)

    results = []
    missing_days = []
    missing_lock = None

    missing_lock = threading.Lock()

    def process_day(day):
        year = day.year
        month = f"{day.month:02d}"
        directory = f"{BASE_DIR}/{year}/{month}/"

        response = http_get(directory, log_name="IMF DSCOVR", timeout=15)
        if response is None:
            return None, True

        html = response.text

        day_str = day.strftime("%Y%m%d000000")
        pattern = rf"(oe_m1m_dscovr_s{day_str}_e\d+_p\d+_pub\.nc\.gz)"
        matches = re.findall(pattern, html)

        if not matches:
            return None, True

        filename = matches[0]
        file_url = directory + filename

        response = http_get(file_url, log_name="IMF DSCOVR", timeout=20)
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_day, day): day for day in days}
        for future in as_completed(future_map):
            day = future_map[future]
            try:
                df, missing = future.result()
                if missing:
                    with missing_lock:
                        missing_days.append(day)
                if df is not None:
                    results.append(df)
            except Exception as exc:
                print(f"[ERROR] Day {format_date(day)} failed: {exc}")

    _emit_missing_ranges("M1M", missing_days)

    if not results:
        return pd.DataFrame(columns=M1M_COLUMNS)

    combined = pd.concat(results).sort_values("time_tag").reset_index(drop=True)
    combined["source_type"] = "archive"

    if BUILD_FROM_REALTIME and REALTIME_BACKFILL_DAYS > 0 and end_date >= start_date:
        realtime_start = end_date - timedelta(days=REALTIME_BACKFILL_DAYS - 1)
        start_dt = datetime.combine(realtime_start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        realtime_df = _download_swpc_mag(start_dt, end_dt)
        if not realtime_df.empty:
            combined = pd.concat([combined, realtime_df], ignore_index=True)
            combined["time_tag"] = pd.to_datetime(combined["time_tag"], errors="coerce", utc=True)
            combined = combined.dropna(subset=["time_tag"]).sort_values("time_tag")
            combined = combined.drop_duplicates(subset="time_tag", keep="last").reset_index(drop=True)

    return combined


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



def _extract(df):
    required = ["time", "bt", "bx_gsm", "by_gsm", "bz_gsm"]

    for col in required:
        if col not in df.columns:
            df[col] = None

    df = df[required].copy()
    df.rename(
        columns={"time": "time_tag", "bx_gsm": "bx", "by_gsm": "by", "bz_gsm": "bz"},
        inplace=True,
    )
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    return df


def _download_swpc_mag(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        resp = requests.get(SWPC_MAG_URL, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return pd.DataFrame(columns=M1M_COLUMNS)

    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=M1M_COLUMNS)

    if isinstance(payload[0], list):
        header = payload[0]
        rows = payload[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.DataFrame(payload)

    lower = {col.lower(): col for col in df.columns}
    time_col = lower.get("time_tag") or lower.get("time") or lower.get("timestamp")
    if time_col is None:
        return pd.DataFrame(columns=M1M_COLUMNS)

    rename = {time_col: "time_tag"}
    mapping = {
        "bt": "bt",
        "bx_gsm": "bx",
        "by_gsm": "by",
        "bz_gsm": "bz",
    }
    for src, dst in mapping.items():
        if src in lower:
            rename[lower[src]] = dst
        else:
            df[dst] = None

    df = df.rename(columns=rename)
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    for col in ("bt", "bx", "by", "bz"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_tag"])
    df = df[(df["time_tag"] >= start_dt) & (df["time_tag"] <= end_dt)]
    df["source_type"] = "realtime"
    return df[M1M_COLUMNS].reset_index(drop=True)
