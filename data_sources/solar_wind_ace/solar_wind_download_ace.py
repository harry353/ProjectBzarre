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

ACE_BASE_URL = (
    "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swepam/level_2_cdaweb/swe_h0"
)
COLUMNS = ["time_tag", "density", "speed", "temperature"]


def download_solar_wind_ace(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download ACE/SWEPAM proton parameters for the provided range.
    """
    if start_date > end_date:
        return pd.DataFrame(columns=COLUMNS)

    frames = []
    current = start_date
    while current <= end_date:
        df = _fetch_day(current)
        if df is not None and not df.empty:
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    data = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    mask = (data["time_tag"].dt.date >= start_date) & (
        data["time_tag"].dt.date <= end_date
    )
    return data.loc[mask].reset_index(drop=True)


def _fetch_day(day: date) -> Optional[pd.DataFrame]:
    resolved = _resolve_filename(day)
    if resolved is None:
        print(f"[INFO] No ACE/SWEPAM data for {format_date(day)}")
        return None

    response = http_get(
        resolved,
        timeout=60,
        log_name="Solar Wind ACE",
        allowed_statuses={404},
    )

    if response is None or response.status_code == 404:
        print(f"[INFO] No ACE/SWEPAM data for {format_date(day)}")
        return None

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
        return df.dropna(subset=["time_tag"])
    finally:
        if cdf is not None and hasattr(cdf, "close"):
            cdf.close()
        os.remove(tmp_path)


def _clean_variable(cdf: cdflib.CDF, variable: str) -> np.ndarray:
    values = np.asarray(cdf.varget(variable), dtype=float)
    attrs = cdf.varattsget(variable)
    fill_value = attrs.get("FILLVAL")
    if fill_value is not None:
        values = np.where(np.isclose(values, float(fill_value)), np.nan, values)
    return values


def _resolve_filename(day: date) -> Optional[str]:
    directory = f"{ACE_BASE_URL}/{day.year}/"
    response = http_get(
        directory,
        timeout=60,
        log_name="Solar Wind ACE",
        allowed_statuses={404},
    )
    if response is None or response.status_code == 404:
        return None

    pattern = re.compile(rf"ac_h0_swe_{day:%Y%m%d}_v(\d{{2}})\.cdf", re.IGNORECASE)
    match = pattern.search(response.text)
    if not match:
        return None
    return f"{directory}{match.group(0)}"
