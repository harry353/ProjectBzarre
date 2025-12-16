from __future__ import annotations

import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from io import BytesIO
from typing import List, Optional

import pandas as pd
import xarray as xr
from common.http import http_get
from space_weather_api import format_date

BASE_DIR = "https://www.ngdc.noaa.gov/dscovr/data"
COLUMNS = ["time_tag", "density", "speed", "temperature"]


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

    def process_day(day):
        day_df = _download_day(day)
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

    if not results:
        return pd.DataFrame(columns=COLUMNS)

    combined = pd.concat(results).sort_values("time_tag").reset_index(drop=True)
    return combined


def _download_day(day) -> Optional[pd.DataFrame]:
    directory = f"{BASE_DIR}/{day.year}/{day.month:02d}/"

    response = http_get(directory, log_name="Solar Wind DSCOVR", timeout=15)
    if response is None:
        return None
    html = response.text

    day_str = day.strftime("%Y%m%d000000")
    pattern = rf"(oe_f1m_dscovr_s{day_str}_e\d+_p\d+_pub\.nc\.gz)"
    matches = re.findall(pattern, html)

    if not matches:
        print(f"[INFO] No DSCOVR F1M match for {format_date(day)}")
        return None

    filename = matches[0]
    file_url = directory + filename

    response = http_get(file_url, log_name="Solar Wind DSCOVR", timeout=20)
    if response is None:
        return None
    gz_bytes = response.content

    try:
        with gzip.open(BytesIO(gz_bytes), "rb") as fh:
            nc_bytes = fh.read()
    except Exception as exc:
        print(f"[WARN] Could not decompress {filename}: {exc}")
        return None

    try:
        ds = xr.open_dataset(BytesIO(nc_bytes), engine="scipy")
    except Exception as exc:
        print(f"[WARN] Failed to open NC file {filename}: {exc}")
        return None

    df = ds.to_dataframe().reset_index()
    ds.close()
    return _extract(df)


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
