from datetime import date
from typing import Iterator, Optional

import pandas as pd
import requests
import re

from common.http import http_get

DST_FINAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_final"
DST_PROVISIONAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional"
DST_REALTIME_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime"
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

    frames = []
    for month_start in _month_range(start_date, end_date):
        if month_start <= date(2004, 12, 31):
            df = _fetch_month_html(month_start, session)
        else:
            base_url = _base_for_month(month_start)
            url = _build_month_url(base_url, month_start)
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


def _base_for_month(month_date: date) -> str:
    if month_date < date(2020, 1, 1):
        return DST_FINAL_BASE
    if month_date <= date(2025, 6, 30):
        return DST_PROVISIONAL_BASE
    return DST_REALTIME_BASE


def _build_month_url(base_url: str, month_date: date) -> str:
    year = month_date.year
    suffix = f"{str(year)[2:]}{month_date.month:02d}.for.request"
    if base_url == DST_REALTIME_BASE and month_date == date.today().replace(day=1):
        return f"{base_url}/presentmonth/dst{suffix}"
    return f"{base_url}/{year}{month_date.month:02d}/dst{suffix}"


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


def _fetch_month_html(month_start: date, session: requests.Session):
    url = f"{DST_FINAL_BASE}/{month_start.year}{month_start.month:02d}/index.html"
    resp = http_get(url, session=session, log_name="DstHTML", timeout=60)
    if resp is None:
        return None

    capture = False
    rows = []
    for line in resp.text.splitlines():
        if "<pre" in line:
            capture = True
            continue
        if "</pre" in line and capture:
            break
        if not capture:
            continue

        matches = re.findall(r"-?\d+", line)
        if len(matches) < 25:
            continue
        day = int(matches[0])
        if not 1 <= day <= 31:
            continue
        values = [int(token) for token in matches[1:25]]
        for hour, dst in enumerate(values[:24]):
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
