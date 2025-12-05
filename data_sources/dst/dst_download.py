from datetime import date
from typing import Iterator, Optional

import pandas as pd
import requests

from common.http import http_get

BASE_RT = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime"
BASE_FINAL = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional"
DST_COLUMNS = ["time_tag", "dst"]


def download_dst(
    start_date: date, end_date: date, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """
    Download hourly Dst index readings for the provided date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    session = session or requests.Session()
    today = date.today()

    frames = []
    for month_start in _month_range(start_date, end_date):
        url = _select_url_for_month(month_start, today)
        df = _fetch_month(url, month_start, session)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=DST_COLUMNS)

    data = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    mask = (data["time_tag"].dt.date >= start_date) & (
        data["time_tag"].dt.date <= end_date
    )

    data = data.loc[mask].reset_index(drop=True)
    data = data.reindex(columns=DST_COLUMNS)
    return data


def _month_range(start: date, end: date) -> Iterator[date]:
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _select_url_for_month(month_date: date, today: date) -> str:
    boundary = date(today.year, today.month, 1)
    for _ in range(4):
        if boundary.month == 1:
            boundary = date(boundary.year - 1, 12, 1)
        else:
            boundary = date(boundary.year, boundary.month - 1, 1)

    year = month_date.year
    suffix = f"{str(year)[2:]}{month_date.month:02d}.for.request"

    if month_date >= boundary:
        if month_date.year == today.year and month_date.month == today.month:
            return f"{BASE_RT}/presentmonth/dst{suffix}"
        return f"{BASE_RT}/{year}{month_date.month:02d}/dst{suffix}"

    return f"{BASE_FINAL}/{year}{month_date.month:02d}/dst{suffix}"


def _fetch_month(url: str, month_start: date, session: requests.Session):
    resp = http_get(url, session=session, log_name="Dst", timeout=60)
    if resp is None:
        return None

    rows = []
    for line in resp.text.splitlines():
        if not line.startswith("DST"):
            continue

        parts = _tokenize_line(line)
        if parts is None:
            continue

        day, values = parts
        for hour, raw_value in enumerate(values):
            dst = _parse_dst_value(raw_value)
            if dst is None:
                continue
            rows.append({"day": day, "hour": hour, "dst": dst})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["year"] = month_start.year
    df["month"] = month_start.month
    df["time_tag"] = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
        ),
        errors="coerce",
    )
    df = df.dropna(subset=["time_tag"])
    return df[["time_tag", "dst"]]


def _tokenize_line(line: str):
    try:
        day = int(line.split("*")[1][:2])
    except Exception:
        return None

    parts = _fix_glued_negatives(line).split()
    if len(parts) < 3:
        return None

    return day, parts[2:]


def _fix_glued_negatives(line: str) -> str:
    fixed = []
    prev = " "
    for ch in line:
        if ch == "-" and prev not in [" ", "-"]:
            fixed.append(" ")
        fixed.append(ch)
        prev = ch
    return "".join(fixed)


def _parse_dst_value(token: str):
    try:
        value = int(token)
    except Exception:
        return None
    if -2000 <= value <= 2000:
        return value
    return None
