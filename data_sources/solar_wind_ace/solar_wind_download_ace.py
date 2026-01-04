from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
import re
from typing import Optional

import cdflib
import numpy as np
import pandas as pd

from common.http import http_get
from space_weather_api import format_date
from database_builder.logging_utils import stamp

ACE_H0_BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0"
ACE_K1_BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_k1"
ACE_K1_SWITCH = date(2024, 7, 1)
COLUMNS = ["time_tag", "density", "speed", "temperature"]


def download_solar_wind_ace(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download ACE/SWEPAM proton parameters for the provided range.
    """
    if start_date > end_date:
        return pd.DataFrame(columns=COLUMNS)

    frames = []
    missing_days = []
    current = start_date
    while current <= end_date:
        df, missing = _fetch_day(current)
        if missing:
            missing_days.append(current)
        if df is not None and not df.empty:
            frames.append(df)
        current += timedelta(days=1)

    _emit_missing_ranges("ACE/SWEPAM", missing_days)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    data = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    mask = (data["time_tag"].dt.date >= start_date) & (
        data["time_tag"].dt.date <= end_date
    )
    return data.loc[mask].reset_index(drop=True)


def _fetch_day(day: date) -> tuple[Optional[pd.DataFrame], bool]:
    resolved = _resolve_filename(day)
    if resolved is None:
        return None, True

    response = http_get(
        resolved,
        timeout=60,
        log_name="Solar Wind ACE",
        allowed_statuses={404},
    )

    if response is None or response.status_code == 404:
        return None, True

    with tempfile.NamedTemporaryFile(suffix=".cdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    cdf = None
    try:
        cdf = cdflib.CDF(tmp_path)
        epoch = cdf.varget("Epoch")
        times = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch), utc=True)

        df = pd.DataFrame(
            {
                "time_tag": times,
                "density": _clean_variable(cdf, "Np"),
                "speed": _clean_variable(cdf, "Vp"),
                "temperature": _clean_variable(cdf, "Tpr"),
            }
        )
        return df.dropna(subset=["time_tag"]), False
    finally:
        if cdf is not None and hasattr(cdf, "close"):
            cdf.close()
        os.remove(tmp_path)


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
            print(
                stamp(
                    f"[WARN] No {label} data for {format_date(start)} -> {format_date(end)}"
                )
            )


def _clean_variable(cdf: cdflib.CDF, variable: str) -> np.ndarray:
    values = np.asarray(cdf.varget(variable), dtype=float)
    attrs = cdf.varattsget(variable)
    fill_value = attrs.get("FILLVAL")
    if fill_value is not None:
        values = np.where(np.isclose(values, float(fill_value)), np.nan, values)
    return values


def _resolve_filename(day: date) -> Optional[str]:
    if day >= ACE_K1_SWITCH:
        base_url = ACE_K1_BASE_URL
        prefix = "ac_k1_swe"
    else:
        base_url = ACE_H0_BASE_URL
        prefix = "ac_h0_swe"

    directory = f"{base_url}/{day.year}/"
    response = http_get(
        directory,
        timeout=60,
        log_name="Solar Wind ACE",
        allowed_statuses={404},
    )
    if response is None or response.status_code == 404:
        return None

    pattern = re.compile(rf"{prefix}_{day:%Y%m%d}_v(\d{{2}})\.cdf", re.IGNORECASE)
    match = pattern.search(response.text)
    if not match:
        return None
    return f"{directory}{match.group(0)}"
