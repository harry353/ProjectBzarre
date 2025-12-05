import re
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from common.http import http_get

AE_MIN = -20000
AE_MAX = 20000
HEADER_RE = re.compile(r"(\d{2})(\d{2})(\d{2})([ELOU])(\d{2})")
BASE_URL = "https://wdc.kugi.kyoto-u.ac.jp/ae_realtime/data_dir"
AE_COLUMNS = ["time_tag", "al", "au", "ae", "ao"]


def download_ae_indices(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Download real-time AL/AU/AE/AO series for the provided date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    session = session or requests.Session()
    frames = []
    current = start_date

    while current <= end_date:
        df = _fetch_al_au_day(current, session)
        if df is not None and not df.empty:
            frames.append(df)
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=AE_COLUMNS)

    df = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    mask = (df["time_tag"].dt.date >= start_date) & (
        df["time_tag"].dt.date <= end_date
    )
    df = df.loc[mask].reset_index(drop=True)
    return df.reindex(columns=AE_COLUMNS)


def _fetch_al_au_day(day: date, session: requests.Session):
    base = f"{BASE_URL}/{day.year}/{day.month:02d}/{day.day:02d}"
    short = f"{str(day.year)[2:]}{day.month:02d}{day.day:02d}"

    urls = {
        "al": f"{base}/al{short}",
        "au": f"{base}/au{short}",
    }

    frames = []
    for component, url in urls.items():
        df = _parse_realtime_ae_file(url, component, session)
        if df is not None:
            frames.append(df)

    if not frames:
        return None

    data = frames[0]
    for df in frames[1:]:
        data = pd.merge(data, df, on="time_tag", how="outer")

    data = data.sort_values("time_tag").reset_index(drop=True)
    data["ae"] = data["au"] - data["al"]
    data["ao"] = (data["au"] + data["al"]) / 2
    return data


def _parse_realtime_ae_file(url: str, component: str, session: requests.Session):
    response = http_get(url, session=session, log_name="AE", timeout=60)
    if response is None:
        return None

    rows = []
    for raw in response.text.splitlines():
        if not raw.startswith("AEALAOAU"):
            continue

        match = HEADER_RE.search(raw)
        if not match:
            continue

        year = 2000 + int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        hour = int(match.group(5))

        base_time = datetime(year, month, day, hour, 0)

        if "QUICKLK" not in raw:
            continue

        values = _extract_values(raw)
        for index, value in enumerate(values):
            timestamp = base_time + timedelta(minutes=index)
            rows.append({"time_tag": timestamp, component: value})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["time_tag"] = pd.to_datetime(df["time_tag"])
    return df


def _extract_values(line: str):
    after = line.split("QUICKLK", 1)[1]
    tokens = after.split()

    values = []
    for token in tokens[:60]:
        values.append(_parse_value(token))

    return values


def _parse_value(token: str):
    try:
        value = int(token)
    except Exception:
        return None

    if AE_MIN <= value <= AE_MAX:
        return value
    return None
