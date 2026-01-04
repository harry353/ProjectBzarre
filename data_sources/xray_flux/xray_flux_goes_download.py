import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from common.http import http_get
from space_weather_api import format_date
from database_builder.logging_utils import stamp


GOES_OPERATIONAL_WINDOWS = {
    "goes16": (date(2017, 2, 7), date(2025, 4, 6)),
    "goes17": (date(2018, 6, 1), date(2023, 1, 1)),
    "goes18": (date(2022, 9, 2), None),
}

BASE_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
FLOAT_COLUMNS = [
    "irradiance_xrsa1",
    "irradiance_xrsa2",
    "irradiance_xrsb1",
    "irradiance_xrsb2",
    "xrs_ratio",
]
FLAG_COLUMNS = ["primary_xrsa", "primary_xrsb"]


def select_satellite(day):
    if isinstance(day, datetime):
        day = day.date()

    available = []

    for sat, (start, end) in GOES_OPERATIONAL_WINDOWS.items():
        if day >= start and (end is None or day <= end):
            available.append(sat)

    if not available:
        raise ValueError(f"No GOES satellite is operational on {format_date(day)}")

    # Sort by satellite number to choose the most recent operational one
    # goes16 < goes17 < goes18
    available.sort(key=lambda s: int(s.replace("goes", "")))

    return available[-1]


def build_goes_url(day, product="sci"):
    if isinstance(day, datetime):
        day = day.date()

    yyyy = day.year
    mm = f"{day.month:02d}"
    dd = f"{day.day:02d}"

    satellite = select_satellite(day)
    sat_num = satellite.replace("goes", "")

    product = (product or "sci").lower()
    if product not in {"sci", "ops"}:
        raise ValueError("product must be 'sci' or 'ops'")

    if product == "sci":
        filename = f"sci_exis-l1b-sfxr_g{sat_num}_d{yyyy}{mm}{dd}_v0-0-1.nc"
        base = (
            "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
            f"{satellite}/l1b/exis-l1b-sfxr_science"
        )
    else:
        filename = f"ops_exis-l1b-sfxr_g{sat_num}_d{yyyy}{mm}{dd}_v0-0-0.nc"
        base = (
            "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
            f"{satellite}/l1b/exis-l1b-sfxr"
        )

    url = f"{base}/{yyyy}/{mm}/{filename}"
    return url, filename


def _cpu_half():
    cores = os.cpu_count() or 1
    return max(1, cores // 2)


def download_xrs_goes(day, dest_folder=".", product="sci"):
    url, filename = build_goes_url(day, product=product)
    dest_path = os.path.join(dest_folder, filename)
    day_str = format_date(day)

    downloaded_here = False
    if not os.path.exists(dest_path):
        try:
            result = _download_file(url, dest_path)
            if result is None:
                return pd.DataFrame()
            downloaded_here = True
        except Exception:
            return pd.DataFrame()

    try:
        df = _resample_to_1min(dest_path)
        if df.empty:
            return df
        if df.index.name != "time_tag" and "time_tag" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        if downloaded_here and os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass


def download_xrs_goes_parallel(
    days: Sequence, dest_folder=".", product="sci", *, emit_missing: bool = True
) -> List[Tuple[date, pd.DataFrame]]:
    """
    Download multiple days in parallel using half the available CPU cores.
    """
    days = list(days)
    if not days:
        return []

    def _job(day):
        try:
            return day, download_xrs_goes(day, dest_folder=dest_folder, product=product)
        except Exception as exc:
            print(f"[WARN] XRS download failed for {format_date(day)}: {exc}")
            return day, pd.DataFrame()

    workers = _cpu_half()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_job, days))

    if emit_missing:
        missing_days = [day for day, df in results if df is None or df.empty]
        _emit_missing_ranges("X Ray Flux GOES", missing_days)
    return results


def _emit_missing_ranges(label: str, days: list[date]) -> None:
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
            print(stamp(f"[WARN] No {label} data for {format_date(start)}"))
        else:
            print(stamp(f"[WARN] No {label} data for {format_date(start)} -> {format_date(end)}"))


def _download_file(url, dest_path):
    response = http_get(
        url,
        log_name="XRay Flux",
        stream=True,
        timeout=30,
        raise_for_status=False,
    )
    if response is None:
        raise RuntimeError(f"Failed to download {url}")

    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        print(f"[WARN] [XRay Flux] Request failed for {response.url}: HTTP {response.status_code}")
        return None

    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    finally:
        response.close()
    return dest_path


def _read_variable(ds: h5py.File, name: str) -> np.ndarray:
    arr = ds[name][:]
    fill = ds[name].attrs.get("_FillValue")
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    return arr


def _resample_to_1min(nc_path):
    with h5py.File(nc_path, "r") as ds:
        seconds = _read_variable(ds, "time")
        float_columns = {name: _read_variable(ds, name) for name in FLOAT_COLUMNS}
        flag_columns = {name: _read_variable(ds, name) for name in FLAG_COLUMNS}

    buckets: Dict[datetime, Dict[str, Dict[str, float]]] = {}

    for idx, sec in enumerate(seconds):
        if np.isnan(sec):
            continue
        minute = _to_minute_timestamp(float(sec))
        bucket = buckets.setdefault(minute, {"sums": defaultdict(float), "counts": defaultdict(int), "flags": {}})

        for name, arr in float_columns.items():
            val = arr[idx]
            if np.isnan(val):
                continue
            bucket["sums"][name] += float(val)
            bucket["counts"][name] += 1

        for name, arr in flag_columns.items():
            val = arr[idx]
            if np.isnan(val):
                continue
            bucket["flags"][name] = int(val)

    rows: List[Dict[str, float]] = []
    for minute in sorted(buckets.keys()):
        bucket = buckets[minute]
        row: Dict[str, float] = {"time_tag": minute}
        for name in FLOAT_COLUMNS:
            count = bucket["counts"].get(name, 0)
            row[name] = bucket["sums"].get(name, 0.0) / count if count else None
        for name in FLAG_COLUMNS:
            row[name] = bucket["flags"].get(name)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "time_tag" not in df.columns:
        return pd.DataFrame()
    df = df.set_index("time_tag").sort_index()
    return df


def _to_minute_timestamp(seconds: float) -> datetime:
    dt = BASE_EPOCH + timedelta(seconds=float(seconds))
    return dt.replace(second=0, microsecond=0)


def main():
    # Example
    day = date(2025, 3, 15)

    sat = select_satellite(day)
    print(f"Selected satellite: {sat}")

    df = download_xrs_goes(day, ".")
    print(f"Rows downloaded: {len(df)}")


if __name__ == "__main__":
    main()
