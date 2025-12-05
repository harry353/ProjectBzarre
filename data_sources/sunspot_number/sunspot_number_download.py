from datetime import date
from typing import Optional

import pandas as pd
import requests

from common.http import http_get

BASE_URL = "https://kp.gfz.de/app/json/"
SN_COLUMNS = ["time_tag", "sunspot_number"]


def _format_day(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def download_sunspot_numbers(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Fetch daily sunspot numbers (SN) from the GFZ API for the provided range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    start = _format_day(start_date)
    end = _format_day(end_date)
    url = f"{BASE_URL}?start={start}T00:00:00Z&end={end}T23:59:00Z&index=SN"

    session = session or requests.Session()
    response = http_get(url, session=session, log_name="Sunspot Number", timeout=30)
    if response is None:
        return pd.DataFrame(columns=SN_COLUMNS)

    payload = response.json()
    times = payload.get("datetime", [])
    values = payload.get("SN", [])

    if not times or not values:
        return pd.DataFrame(columns=SN_COLUMNS)

    df = pd.DataFrame(
        {
            "time_tag": pd.to_datetime(times, utc=True, errors="coerce"),
            "sunspot_number": pd.to_numeric(values, errors="coerce"),
        }
    )

    df = df.dropna(subset=["time_tag"]).reset_index(drop=True)
    df = df.sort_values("time_tag").reset_index(drop=True)
    return df.reindex(columns=SN_COLUMNS)
