import gzip
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from io import BytesIO

import pandas as pd
import xarray as xr

from common.http import http_get

from space_weather_api import format_date

BASE_DIR = "https://www.ngdc.noaa.gov/dscovr/data"
M1M_COLUMNS = ["time_tag", "bt", "bx", "by", "bz"]


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

    def process_day(day):
        year = day.year
        month = f"{day.month:02d}"
        directory = f"{BASE_DIR}/{year}/{month}/"

        response = http_get(directory, log_name="IMF DSCOVR", timeout=15)
        if response is None:
            return None

        html = response.text

        day_str = day.strftime("%Y%m%d000000")
        pattern = rf"(oe_m1m_dscovr_s{day_str}_e\d+_p\d+_pub\.nc\.gz)"
        matches = re.findall(pattern, html)

        if not matches:
            print(f"[INFO] No M1M match for {format_date(day)}")
            return None

        filename = matches[0]
        file_url = directory + filename

        response = http_get(file_url, log_name="IMF DSCOVR", timeout=20)
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_day, day): day for day in days}
        for future in as_completed(future_map):
            day = future_map[future]
            try:
                df = future.result()
                if df is not None:
                    results.append(df)
            except Exception as exc:
                print(f"[ERROR] Day {format_date(day)} failed: {exc}")

    if not results:
        return pd.DataFrame(columns=M1M_COLUMNS)

    return pd.concat(results).sort_values("time_tag").reset_index(drop=True)


def _extract(df):
    required = ["time", "bt", "bx_gse", "by_gse", "bz_gse"]

    for col in required:
        if col not in df.columns:
            df[col] = None

    df = df[required].copy()
    df.rename(
        columns={"time": "time_tag", "bx_gse": "bx", "by_gse": "by", "bz_gse": "bz"},
        inplace=True,
    )
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    return df
