"""Download and normalize GOES flare archive NetCDF files."""

from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr

from common.http import http_get
from space_weather_api import format_date

GOES_RANGES: Dict[str, Tuple[str, str]] = {
    "g08": ("1995-01-03", "2003-06-16"),
    "g09": ("1996-04-03", "1998-07-28"),
    "g10": ("1998-07-01", "2009-12-01"),
    "g11": ("2006-06-01", "2008-02-10"),
    "g12": ("2003-01-10", "2007-04-12"),
    "g13": ("2013-06-07", "2017-12-14"),
    "g14": ("2009-09-21", "2019-10-01"),
    "g15": ("2010-04-08", "2020-01-10"),
}

OUTPUT_COLUMNS = [
    "flare_id",
    "event_time",
    "flare_class",
    "peak_flux_wm2",
    "status",
    "xrsb_flux",
    "background_flux",
    "integrated_flux",
    "satellite",
    "source_day",
    "file_url",
]

BASE_DIR = (
    "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs"
)


def determine_goes_satellite(day: date) -> str:
    if isinstance(day, datetime):
        day = day.date()

    for goes, (start, end) in GOES_RANGES.items():
        if pd.to_datetime(start).date() <= day <= pd.to_datetime(end).date():
            return goes
    raise ValueError(f"No GOES satellite defined for {format_date(day)}")


def discover_version(goes: str, day: date) -> Tuple[str, str]:
    d = day if isinstance(day, datetime) else datetime.combine(day, datetime.min.time())
    yyyy = d.year
    mm = f"{d.month:02d}"
    dd = f"{d.day:02d}"

    goes_dir = f"goes{goes[1:]}"
    dir_url = f"{BASE_DIR}/{goes_dir}/xrsf-l2-flsum_science/{yyyy}/{mm}/"

    response = http_get(dir_url, log_name="Flares Archive", timeout=60)
    if response is None:
        raise ValueError(f"Could not read directory listing: {dir_url}")

    pattern = re.compile(
        rf"sci_xrsf-l2-flsum_{goes}_d{yyyy}{mm}{dd}_v\d+-\d+-\d+\.nc", re.IGNORECASE
    )
    matches = pattern.findall(response.text)
    if not matches:
        raise FileNotFoundError(f"No flare summary file for {format_date(day)} ({goes})")

    return dir_url, matches[0]


def build_archive_url(day: date) -> Tuple[str, str, str]:
    goes = determine_goes_satellite(day)
    dir_url, filename = discover_version(goes, day)
    full_url = dir_url + filename
    satellite = f"goes{goes[1:]}"
    return full_url, filename, satellite


def download_flares_archive(start_date: date, end_date: date) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        try:
            frame = _download_day(current)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[WARN] GOES flare archive failed for {format_date(current)}: {exc}")
            frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if not frame.empty:
            frames.append(frame)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["flare_id", "event_time"], keep="first")
    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def _download_day(day: date) -> pd.DataFrame:
    url, filename, satellite = build_archive_url(day)
    response = http_get(url, log_name="Flares Archive", timeout=60)
    if response is None:
        print(f"[WARN] GOES flare archive returned no data for {format_date(day)}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        with xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf") as ds:
            return _dataset_to_frame(ds, day, satellite, url)
    except Exception as exc:
        print(f"[WARN] GOES flare archive failed to parse {filename}: {exc}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _dataset_to_frame(ds: xr.Dataset, day: date, satellite: str, url: str) -> pd.DataFrame:
    count = int(ds.sizes.get("time", 0))
    if count == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    def _safe_values(name: str):
        if name not in ds.variables:
            return [None] * count
        values = ds[name].values
        if values.shape and values.shape[0] == count:
            return values
        return [None] * count

    data = {
        "flare_id": _safe_values("flare_id"),
        "event_time": _safe_values("time"),
        "flare_class": _safe_values("flare_class"),
        "status": _safe_values("status"),
        "xrsb_flux": _safe_values("xrsb_flux"),
        "background_flux": _safe_values("background_flux"),
        "integrated_flux": _safe_values("integrated_flux"),
    }

    df = pd.DataFrame(data)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["flare_id"] = df["flare_id"].apply(_decode_str)
    df["flare_class"] = df["flare_class"].apply(_decode_str)
    df["status"] = df["status"].apply(_decode_str)

    df["peak_flux_wm2"] = df["flare_class"].map(_parse_flare_class)
    df["satellite"] = satellite
    df["source_day"] = day.isoformat()
    df["file_url"] = url

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[OUTPUT_COLUMNS]
    df = df.dropna(subset=["event_time", "flare_id"], how="any")
    return df.reset_index(drop=True)


def _decode_str(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _parse_flare_class(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.upper()
    base_map = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }
    base = base_map.get(value[0])
    if base is None:
        return None
    try:
        magnitude = float(value[1:] or "0")
    except ValueError:
        return None
    return base * magnitude
