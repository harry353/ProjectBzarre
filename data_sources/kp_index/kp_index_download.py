from datetime import date
from typing import Optional

import pandas as pd
import requests

from common.http import http_get

BASE_URL = "https://kp.gfz.de/app/json/"
KP_COLUMNS = ["time_tag", "kp_index"]


def _format_day(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def download_kp_index(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Fetch GFZ Kp index readings for the provided date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    start = _format_day(start_date)
    end = _format_day(end_date)
    url = f"{BASE_URL}?start={start}T00:00:00Z&end={end}T23:59:00Z&index=Kp"

    session = session or requests.Session()
    response = http_get(url, session=session, log_name="Kp Index", timeout=30)
    if response is None:
        return pd.DataFrame(columns=KP_COLUMNS)

    payload = response.json()
    times = payload.get("datetime", [])
    values = payload.get("Kp", [])

    if not times or not values:
        return pd.DataFrame(columns=KP_COLUMNS)

    df = pd.DataFrame(
        {
            "time_tag": pd.to_datetime(times, utc=True, errors="coerce"),
            "kp_index": pd.to_numeric(values, errors="coerce"),
        }
    )
    df = df.dropna(subset=["time_tag"]).reset_index(drop=True)
    df = df.sort_values("time_tag").reset_index(drop=True)
    return df.reindex(columns=KP_COLUMNS)
