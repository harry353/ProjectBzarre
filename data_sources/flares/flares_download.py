"""Download GOES flare summary NetCDF files and normalise into DataFrames."""

from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import xarray as xr

from common.http import http_get
from space_weather_api import format_date

GOES_OPERATIONAL_WINDOWS: Dict[str, Tuple[date, date | None]] = {
    "goes16": (date(2017, 2, 7), date(2025, 4, 6)),
    "goes17": (date(2018, 6, 1), date(2023, 1, 10)),
    "goes18": (date(2022, 9, 2), None),
}

FLARE_VARIABLES = [
    "time",
    "flare_id",
    "flare_class",
    "status",
    "xrsb_flux",
    "background_flux",
    "integrated_flux",
]

OUTPUT_COLUMNS = [
    "flare_id",
    "event_time",
    "flare_class",
    "peak_flux_wm2",
    "status",
    "xrsb_flux",
    "background_flux",
    "integrated_flux",
    "source_day",
    "satellite",
]


def select_satellite(day: date) -> str:
    """Return the operational GOES satellite for the provided day."""

    available: List[str] = []
    for sat, (start, end) in GOES_OPERATIONAL_WINDOWS.items():
        if day >= start and (end is None or day <= end):
            available.append(sat)

    if not available:
        raise ValueError(f"No GOES satellite is operational on {format_date(day)}")

    available.sort(key=lambda s: int(s.replace("goes", "")))
    return available[-1]


def build_flares_url(day: date) -> Tuple[str, str, str]:
    """Construct the NOAA archive URL for a GOES flare summary file."""

    if isinstance(day, datetime):  # defensive
        day = day.date()

    satellite = select_satellite(day)
    sat_token = satellite.replace("goes", "g")

    yyyy = day.year
    mm = f"{day.month:02d}"
    dd = f"{day.day:02d}"

    base = (
        "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
        f"{satellite}/l2/data/xrsf-l2-flsum_science"
    )
    filename = f"sci_xrsf-l2-flsum_{sat_token}_d{yyyy}{mm}{dd}_v2-2-0.nc"
    url = f"{base}/{yyyy}/{mm}/{filename}"
    return url, filename, satellite


def download_flares(start_date: date, end_date: date) -> pd.DataFrame:
    """Download flare summary rows for every day in the inclusive range."""

    frames: List[pd.DataFrame] = []
    current = start_date
    while current <= end_date:
        try:
            frame = _download_day(current)
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[WARN] GOES flares download failed for {format_date(current)}: {exc}")
            frame = pd.DataFrame(columns=OUTPUT_COLUMNS)
        if not frame.empty:
            frames.append(frame)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["flare_id"])
    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def _download_day(day: date) -> pd.DataFrame:
    url, filename, satellite = build_flares_url(day)
    response = http_get(url, log_name="GOES Flares", timeout=60)
    if response is None:
        print(f"[WARN] GOES flares request returned no data for {format_date(day)}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        with xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf") as ds:
            frame = _dataset_to_frame(ds, day, satellite)
    except Exception as exc:
        print(f"[WARN] GOES flares failed to parse {filename}: {exc}")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return frame


def _dataset_to_frame(ds: xr.Dataset, day: date, satellite: str) -> pd.DataFrame:
    count = int(ds.sizes.get("time", 0))
    if count == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    data = {name: _extract_array(ds, name, count) for name in FLARE_VARIABLES}

    df = pd.DataFrame(data)
    df.rename(columns={"time": "event_time"}, inplace=True)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["flare_class"] = df["flare_class"].apply(_decode_str)
    df["status"] = df["status"].apply(_decode_str)
    df["flare_id"] = df["flare_id"].apply(_decode_str)

    df["peak_flux_wm2"] = df["flare_class"].map(_parse_flare_class)
    df["peak_flux_wm2"] = df["peak_flux_wm2"].fillna(df.get("xrsb_flux"))
    df["peak_flux_wm2"] = df["peak_flux_wm2"].fillna(df.get("xrsb_flux"))
    df["source_day"] = day.isoformat()
    df["satellite"] = satellite

    # reorder columns and drop rows missing IDs or times
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLUMNS]
    df = df.dropna(subset=["flare_id", "event_time"])
    return df.reset_index(drop=True)


def _extract_array(ds: xr.Dataset, name: str, count: int):
    if name not in ds.variables:
        return [None] * count
    values = ds[name].values
    if values.shape and values.shape[0] == count:
        return values
    return [None] * count


def _decode_str(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _parse_flare_class(value) -> float | None:
    mapping = {
        "A": 1e-8,
        "B": 1e-7,
        "C": 1e-6,
        "M": 1e-5,
        "X": 1e-4,
    }
    if not value:
        return None
    value = value.upper()
    prefix = value[0]
    base = mapping.get(prefix)
    if base is None:
        return None
    magnitude = value[1:] or "0"
    try:
        number = float(magnitude)
    except ValueError:
        return None
    return base * number
