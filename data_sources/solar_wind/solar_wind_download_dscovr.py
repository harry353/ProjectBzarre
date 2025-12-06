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
    required = ["time", "proton_density", "proton_speed", "proton_temperature"]

    for col in required:
        if col not in df.columns:
            df[col] = None

    df = df[required].copy()
    df.rename(
        columns={
            "time": "time_tag",
            "proton_density": "density",
            "proton_speed": "speed",
            "proton_temperature": "temperature",
        },
        inplace=True,
    )

    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    return df
