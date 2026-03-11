from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from common.http import http_get
from database_builder.constants import BUILD_FROM_REALTIME, REALTIME_BACKFILL_DAYS

BASE_URL = "https://kp.gfz.de/app/json/"
SWPC_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
KP_COLUMNS = ["time_tag", "kp_index", "source_type"]


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
        df = pd.DataFrame(columns=KP_COLUMNS)
    else:
        df = pd.DataFrame(
            {
                "time_tag": pd.to_datetime(times, utc=True, errors="coerce"),
                "kp_index": pd.to_numeric(values, errors="coerce"),
            }
        )
        df = df.dropna(subset=["time_tag"]).reset_index(drop=True)
        df = df.sort_values("time_tag").reset_index(drop=True)
        df["source_type"] = "archive"

    if BUILD_FROM_REALTIME and REALTIME_BACKFILL_DAYS > 0 and end_date >= start_date:
        realtime_start = end_date - timedelta(days=REALTIME_BACKFILL_DAYS - 1)
        start_dt = datetime.combine(realtime_start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        realtime_df = _fetch_swpc_kp(start_dt, end_dt, session)
        if not realtime_df.empty:
            if df.empty:
                df = realtime_df
            else:
                df = pd.concat([df, realtime_df], ignore_index=True)
                df = df.sort_values("time_tag").drop_duplicates(subset="time_tag", keep="last").reset_index(drop=True)

    mask = (df["time_tag"].dt.date >= start_date) & (df["time_tag"].dt.date <= end_date)
    df = df.loc[mask].reset_index(drop=True)

    return df.reindex(columns=KP_COLUMNS)


def _fetch_swpc_kp(
    start_dt: datetime, end_dt: datetime, session: requests.Session
) -> pd.DataFrame:
    resp = http_get(SWPC_KP_URL, session=session, log_name="Kp SWPC", timeout=30)
    if resp is None:
        return pd.DataFrame(columns=KP_COLUMNS)
    
    try:
        payload = resp.json()
    except Exception:
        return pd.DataFrame(columns=KP_COLUMNS)

    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame(columns=KP_COLUMNS)

    header = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=header)
    
    lower = {col.lower(): col for col in df.columns}
    time_col = lower.get("time_tag")
    kp_col = lower.get("kp")
    
    if time_col is None or kp_col is None:
        return pd.DataFrame(columns=KP_COLUMNS)

    df = df.rename(columns={time_col: "time_tag", kp_col: "kp_index"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df["kp_index"] = pd.to_numeric(df["kp_index"], errors="coerce")
    df = df.dropna(subset=["time_tag", "kp_index"])
    
    df = df[(df["time_tag"] >= start_dt) & (df["time_tag"] <= end_dt)]
    df["source_type"] = "realtime"
    return df[KP_COLUMNS].reset_index(drop=True)
