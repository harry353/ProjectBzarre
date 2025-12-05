from datetime import date, datetime, timedelta
from io import StringIO
from typing import Optional

import pandas as pd
import requests

from common.http import http_get

SUPERMAG_ENDPOINT = "https://supermag.jhuapl.edu/services/indices.php"


def download_supermag_indices(
    start_date: date,
    end_date: date,
    logon: str,
    chunk_hours: int = 24,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download SuperMAG SME/SMU/SML indices for the requested date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")
    if not logon:
        raise ValueError("A SuperMAG logon is required.")
    if chunk_hours <= 0:
        raise ValueError("chunk_hours must be positive.")

    session = session or requests.Session()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
    chunk_delta = timedelta(hours=chunk_hours)

    frames = []
    current = start_dt

    while current < end_dt:
        extent_seconds = int(min(chunk_delta, end_dt - current).total_seconds())
        df = _fetch_supermag_chunk(logon, current, extent_seconds, session)
        if df is not None and not df.empty:
            frames.append(df)
        current += chunk_delta

    if not frames:
        return pd.DataFrame(columns=["time", "SML", "SMU", "SME"])

    combined = pd.concat(frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], errors="coerce")
    combined = combined.dropna(subset=["time"])
    combined = combined.sort_values("time")
    combined = combined.drop_duplicates(subset=["time"])

    mask = (combined["time"] >= start_dt) & (combined["time"] < end_dt)
    return combined.loc[mask].reset_index(drop=True)


def _fetch_supermag_chunk(
    logon: str,
    start_dt: datetime,
    extent_seconds: int,
    session: requests.Session,
) -> pd.DataFrame:
    params = {
        "fmt": "json",
        "logon": logon,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M"),
        "extent": str(extent_seconds),
        "indices": "all",
    }

    response = http_get(
        SUPERMAG_ENDPOINT,
        session=session,
        log_name="SuperMAG",
        params=params,
        timeout=120,
    )
    if response is None:
        return pd.DataFrame()

    text = response.text.lstrip()
    if text.startswith("OK"):
        text = text[2:].lstrip()

    if not text.strip():
        return pd.DataFrame()

    try:
        df = pd.read_json(StringIO(text))
    except ValueError as exc:
        print(f"[WARN] Invalid JSON payload from SuperMAG: {exc}")
        return pd.DataFrame()

    if "tval" not in df:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["tval"], unit="s", errors="coerce")
    df = df.dropna(subset=["time"])
    return df
