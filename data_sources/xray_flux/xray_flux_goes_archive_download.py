import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from common.http import http_get
from space_weather_api import format_date


ARCHIVE_WINDOWS = [
    ("goes08", date(1995, 1, 3), date(2003, 6, 16)),
    ("goes09", date(1996, 4, 1), date(1998, 7, 28)),
    ("goes10", date(1998, 7, 1), date(2009, 12, 1)),
    ("goes11", date(2006, 6, 1), date(2008, 2, 10)),
    ("goes12", date(2003, 1, 10), date(2007, 4, 12)),
    ("goes13", date(2013, 6, 7), date(2017, 12, 14)),
    ("goes14", date(2019, 9, 19), date(2020, 3, 4)),
    ("goes15", date(2010, 4, 7), date(2020, 3, 4)),
]

ARCHIVE_BASE = (
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs"
)


def _cpu_half():
    cores = os.cpu_count() or 1
    return max(1, cores // 2)


def select_archive_satellite(day: date) -> Tuple[str, str]:
    if isinstance(day, datetime):
        day = day.date()

    for sat, start, end in ARCHIVE_WINDOWS:
        if day >= start and day <= end:
            return sat, sat.replace("goes", "")

    raise ValueError(f"No GOES archive satellite covers {format_date(day)}")


def _list_month_files(satellite: str, year: int, month: int) -> List[str]:
    key = (satellite, year, month)
    cache = _list_month_files.cache
    if key in cache:
        return cache[key]

    url = f"{ARCHIVE_BASE}/{satellite}/xrsf-l2-avg1m_science/{year}/{month:02d}/"
    response = http_get(url, log_name="XRay Flux Archive", timeout=30, raise_for_status=False)
    if response is None or response.status_code == 404:
        cache[key] = []
        return []

    text = response.text
    files = re.findall(r'href="([^" ]+\.nc)"', text)
    cache[key] = files
    return files


_list_month_files.cache: Dict[Tuple[str, int, int], List[str]] = {}


def _find_archive_filename(satellite: str, sat_num: str, day: date) -> Optional[str]:
    files = _list_month_files(satellite, day.year, day.month)
    if not files:
        return None

    pattern = re.compile(
        rf"sci_xrsf-l2-avg1m_g{sat_num}_d{day.year}{day.month:02d}{day.day:02d}_v[0-9\-]+\.nc",
        re.IGNORECASE,
    )

    matches = sorted([f for f in files if pattern.fullmatch(f)])
    if not matches:
        return None

    return matches[-1]


def download_xrs_goes_archive(day, dest_folder="."):
    if isinstance(day, datetime):
        day = day.date()

    try:
        satellite, sat_num = select_archive_satellite(day)
    except ValueError:
        print(f"No archive satellite for {format_date(day)}")
        return pd.DataFrame()

    filename = _find_archive_filename(satellite, sat_num, day)
    if not filename:
        print(f"No archive file found for {format_date(day)}")
        return pd.DataFrame()

    base = f"{ARCHIVE_BASE}/{satellite}/xrsf-l2-avg1m_science/{day.year}/{day.month:02d}"
    url = f"{base}/{filename}"
    dest_path = os.path.join(dest_folder, filename)

    downloaded_here = False
    if not os.path.exists(dest_path):
        response = http_get(url, log_name="XRay Flux Archive", stream=True, timeout=60)
        if response is None:
            print(f"Failed to download archive file for {format_date(day)}")
            return pd.DataFrame()

        try:
            with open(dest_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
            downloaded_here = True
        finally:
            response.close()

    try:
        return _resample_archive_file(dest_path)
    except Exception as exc:
        print(f"[WARN] Failed to process archive file {dest_path}: {exc}")
        return pd.DataFrame()
    finally:
        if downloaded_here and os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass


def download_xrs_goes_archive_parallel(
    days: Sequence, dest_folder="."
) -> List[Tuple[date, pd.DataFrame]]:
    days = list(days)
    if not days:
        return []

    def _job(single_day):
        try:
            return single_day, download_xrs_goes_archive(single_day, dest_folder=dest_folder)
        except Exception as exc:
            print(f"[WARN] Archive XRS download failed for {format_date(single_day)}: {exc}")
            return single_day, pd.DataFrame()

    with ThreadPoolExecutor(max_workers=_cpu_half()) as executor:
        return list(executor.map(_job, days))


def _resample_archive_file(nc_path: str) -> pd.DataFrame:
    with h5py.File(nc_path, "r") as ds:
        time_var = ds["time"][:]
        time_attrs = dict(ds["time"].attrs)
        xrsa = ds["xrsa_flux"][:]
        xrsb = ds["xrsb_flux"][:]
        xrsa_flags = ds.get("xrsa_flags")
        xrsb_flags = ds.get("xrsb_flags")

    timestamps = _convert_archive_times(time_var, time_attrs)

    primary_xrsa = np.zeros_like(xrsa, dtype=int)
    primary_xrsb = np.zeros_like(xrsb, dtype=int)

    df = pd.DataFrame(
        {
            "time_tag": timestamps,
            "irradiance_xrsa1": xrsa,
            "irradiance_xrsa2": xrsa,
            "irradiance_xrsb1": xrsb,
            "irradiance_xrsb2": xrsb,
            "xrs_ratio": np.divide(
                xrsb,
                xrsa,
                out=np.full_like(xrsb, np.nan, dtype=float),
                where=~np.isnan(xrsa) & (xrsa != 0),
            ),
            "primary_xrsa": primary_xrsa,
            "primary_xrsb": primary_xrsb,
        }
    )

    df = df.dropna(subset=["time_tag"])
    df = df.set_index("time_tag")
    return df.sort_index()


def _convert_archive_times(values: np.ndarray, attrs) -> List[Optional[datetime]]:
    base = _parse_time_units(attrs)
    if base is None:
        base = datetime(1970, 1, 1)

    timestamps: List[Optional[datetime]] = []
    for raw in values:
        if np.isnan(raw):
            timestamps.append(pd.NaT)
            continue
        ts = base + timedelta(seconds=float(raw))
        timestamps.append(ts.replace(second=0, microsecond=0))
    return timestamps


def _parse_time_units(attrs) -> Optional[datetime]:
    units = attrs.get("units") if attrs else None
    if isinstance(units, bytes):
        units = units.decode()

    if isinstance(units, str):
        match = re.search(r"seconds since ([0-9]{4}-[0-9]{2}-[0-9]{2})(?:[ T]([0-9:]+))?", units)
        if match:
            date_part = match.group(1)
            time_part = match.group(2) or "00:00:00"
            try:
                return datetime.fromisoformat(f"{date_part}T{time_part}")
            except ValueError:
                pass
    return None
