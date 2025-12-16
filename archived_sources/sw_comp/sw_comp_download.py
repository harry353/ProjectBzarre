import os
import tempfile
from datetime import date, timedelta
from io import BytesIO
from typing import Dict, Optional
import re

import cdflib
import pandas as pd
import requests

from common.http import http_get

BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swics/level_2_cdaweb/sw2_h3"
LEGACY_BASE_URL = (
    "https://cdaweb.gsfc.nasa.gov/pub/data/ace/swics/level_2_cdaweb/swi_h2"
)
COMPOSITION_VARS: Dict[str, str] = {
    "O7to6": "o7_o6",
    "C6to5": "c6_c5",
    "avqFe": "avg_fe_charge",
    "FetoO": "fe_to_o",
}
OUTPUT_COLUMNS = ["time_tag"] + list(COMPOSITION_VARS.values())


def download_sw_comp(
    start_date: date,
    end_date: date,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download ACE/SWICS composition ratios for the provided date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    session = session or requests.Session()
    frames = []
    current = start_date

    while current <= end_date:
        df = _fetch_day(current, session)
        if df is not None and not df.empty:
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    data = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    mask = (data["time_tag"].dt.date >= start_date) & (
        data["time_tag"].dt.date <= end_date
    )
    return data.loc[mask].reset_index(drop=True)


def _fetch_day(day, session: requests.Session) -> Optional[pd.DataFrame]:
    base_url, prefix = _source_for_day(day)
    year = day.year

    filename = _resolve_swics_filename(base_url, prefix, day, session)
    if filename is None:
        print(f"[INFO] No ACE/SWICS composition data for {day:%Y-%m-%d}")
        return None

    url = f"{base_url}/{year}/{filename}"

    response = http_get(
        url,
        session=session,
        log_name="SW COMP",
        timeout=60,
        allowed_statuses={404},
    )
    if response is None:
        return None
    if response.status_code == 404:
        print(f"[INFO] No ACE/SWICS composition data for {day:%Y-%m-%d}")
        return None

    try:
        return _parse_cdf(BytesIO(response.content))
    except Exception as exc:
        print(f"[WARN] Failed to parse {filename}: {exc}")
        return None


def _source_for_day(day: date):
    legacy_boundary = date(2011, 8, 21)
    modern_boundary = date(2015, 1, 1)

    if day < legacy_boundary:
        return LEGACY_BASE_URL, "ac_h2_swi"
    if day < modern_boundary:
        return BASE_URL, "ac_h3_sw2"
    return BASE_URL, "ac_h3_sw2"


def _parse_cdf(handle: BytesIO) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".cdf", delete=False) as tmp:
        tmp.write(handle.getvalue())
        tmp_path = tmp.name

    cdf = None
    try:
        cdf = cdflib.CDF(tmp_path)
        epoch = cdf.varget("Epoch")
        times = cdflib.cdfepoch.to_datetime(epoch)
        data = {"time_tag": pd.to_datetime(times)}
        for variable, alias in COMPOSITION_VARS.items():
            try:
                data[alias] = cdf.varget(variable)
            except Exception:
                data[alias] = None
        df = pd.DataFrame(data).dropna(subset=["time_tag"])
    finally:
        if cdf is not None and hasattr(cdf, "close"):
            cdf.close()
        os.remove(tmp_path)

    return df.reindex(columns=OUTPUT_COLUMNS)


def _resolve_swics_filename(base_url: str, prefix: str, day: date, session: requests.Session) -> Optional[str]:
    directory = f"{base_url}/{day.year}/"
    response = http_get(
        directory,
        session=session,
        log_name="SW COMP",
        timeout=60,
        allowed_statuses={404},
    )
    if response is None or response.status_code == 404:
        return None

    pattern = re.compile(rf"({prefix}_{day:%Y%m%d}_v\d{{2}}\.cdf)", re.IGNORECASE)
    match = pattern.search(response.text)
    if not match:
        return None
    return match.group(1)
