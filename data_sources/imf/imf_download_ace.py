import os
import tempfile
import re
from datetime import date, timedelta
from io import BytesIO
from typing import Optional

import cdflib
import numpy as np
import pandas as pd
import requests

from common.http import http_get

BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/ace/mag/level_2_cdaweb/mfi_h3"
OUTPUT_COLUMNS = ["time_tag", "bx_gse", "by_gse", "bz_gse", "bt"]
FILL_VALUE = -1e20


def download_imf_ace(
    start_date: date,
    end_date: date,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download ACE/MAG IMF series for the requested date range.
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


def _fetch_day(day, session: requests.Session):
    filename = _resolve_filename(day, session)
    if filename is None:
        print(f"[INFO] No ACE/MAG data found for {day:%Y-%m-%d}")
        return None

    url = f"{BASE_URL}/{day.year}/{filename}"
    response = http_get(
        url,
        session=session,
        log_name="IMF ACE",
        timeout=60,
        allowed_statuses={404},
    )
    if response is None or response.status_code == 404:
        print(f"[INFO] No ACE/MAG data found for {day:%Y-%m-%d}")
        return None

    try:
        return _parse_cdf(BytesIO(response.content))
    except Exception as exc:
        print(f"[WARN] Failed to parse {filename}: {exc}")
        return None


def _parse_cdf(handle: BytesIO) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".cdf", delete=False) as tmp:
        tmp.write(handle.getvalue())
        tmp_path = tmp.name

    cdf = None
    try:
        cdf = cdflib.CDF(tmp_path)
        epoch = cdf.varget("Epoch")
        times = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch))

        bgse = cdf.varget("BGSEc")
        mag = cdf.varget("Magnitude")

        bgse = np.where(bgse < FILL_VALUE, np.nan, bgse)
        mag = np.where(mag < FILL_VALUE, np.nan, mag)

        df = pd.DataFrame(
            {
                "time_tag": times,
                "bx_gse": bgse[:, 0],
                "by_gse": bgse[:, 1],
                "bz_gse": bgse[:, 2],
                "bt": mag,
            }
        )
        return df.reindex(columns=OUTPUT_COLUMNS)
    finally:
        if cdf is not None and hasattr(cdf, "close"):
            cdf.close()
        os.remove(tmp_path)


def _resolve_filename(day: date, session: requests.Session) -> Optional[str]:
    directory = f"{BASE_URL}/{day.year}/"
    response = http_get(
        directory,
        session=session,
        log_name="IMF ACE",
        timeout=60,
        allowed_statuses={404},
    )
    if response is None or response.status_code == 404:
        return None

    pattern = re.compile(rf"(ac_h3_mfi_{day:%Y%m%d}_v\d{{2}}\.cdf)", re.IGNORECASE)
    match = pattern.search(response.text)
    if not match:
        return None
    return match.group(1)
