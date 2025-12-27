from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Optional

import pandas as pd
import requests
import re

from common.http import http_get
from database_builder.constants import BUILD_FROM_REALTIME, REALTIME_BACKFILL_DAYS

DST_FINAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_final"
DST_PROVISIONAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional"
DST_REALTIME_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime"
DST_COLUMNS = ["time_tag", "dst", "source_type"]
DST_SWPC_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"


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
        base_url = _base_for_month(month_start)
        if month_start <= date(2004, 12, 31) or base_url == DST_REALTIME_BASE:
            df = _fetch_month_html(base_url, month_start, session)
        else:
            url = _build_month_url(base_url, month_start)
            df = _fetch_month(url, month_start, session)
            html_df = _fetch_month_html(base_url, month_start, session)
            if df is None:
                df = html_df
            elif html_df is not None:
                df = _fill_with_html(df, html_df, prefer_html=base_url == DST_REALTIME_BASE)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=DST_COLUMNS)

    data = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    data["time_tag"] = data["time_tag"] + pd.Timedelta(hours=1)
    mask = (data["time_tag"].dt.date >= start_date) & (
        data["time_tag"].dt.date <= end_date
    )
    data = data.loc[mask].reset_index(drop=True)
    data["source_type"] = "archive"
    data["_valid"] = data["dst"].notna().astype(int)
    data = (
        data.sort_values(["time_tag", "_valid"])
        .drop_duplicates(subset="time_tag", keep="last")
        .drop(columns="_valid")
        .reset_index(drop=True)
    )
    if BUILD_FROM_REALTIME and REALTIME_BACKFILL_DAYS > 0:
        realtime_start = end_date - timedelta(days=REALTIME_BACKFILL_DAYS - 1)
        start_dt = datetime.combine(realtime_start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        swpc = _fetch_swpc_dst(start_dt, end_dt, session)
        if not swpc.empty:
            swpc["time_tag"] = swpc["time_tag"] + pd.Timedelta(hours=1)
            left = data.set_index("time_tag")
            right = swpc.set_index("time_tag")
            left.update(right[["dst", "source_type"]])
            data = left.reset_index()
    data = data.dropna(subset=["dst"]).reset_index(drop=True)
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
    if month_date < date(2020, 12, 31):
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


def _build_month_html_url(base_url: str, month_date: date) -> str:
    if base_url == DST_REALTIME_BASE and month_date == date.today().replace(day=1):
        return f"{base_url}/presentmonth/index.html"
    return f"{base_url}/{month_date.year}{month_date.month:02d}/index.html"


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
    df["source_type"] = "realtime"
    return df[["time_tag", "dst", "source_type"]]


def _fetch_month_html(base_url: str, month_start: date, session: requests.Session):
    url = _build_month_html_url(base_url, month_start)
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
        if len(matches) < 2:
            continue
        day = int(matches[0])
        if not 1 <= day <= 31:
            continue
        values = [_parse_dst_value(token) for token in matches[1:]]
        if len(values) < 24:
            values.extend([None] * (24 - len(values)))
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


def _fetch_swpc_dst(
    start_dt: datetime, end_dt: datetime, session: requests.Session
) -> pd.DataFrame:
    resp = http_get(DST_SWPC_URL, session=session, log_name="DstSWPC", timeout=30)
    if resp is None:
        return pd.DataFrame(columns=DST_COLUMNS)
    try:
        payload = resp.json()
    except Exception:
        return pd.DataFrame(columns=DST_COLUMNS)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=DST_COLUMNS)
    if isinstance(payload[0], list):
        header = payload[0]
        rows = payload[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.DataFrame(payload)
    lower = {col.lower(): col for col in df.columns}
    time_col = lower.get("time_tag") or lower.get("time") or lower.get("date")
    dst_col = lower.get("dst")
    if time_col is None or dst_col is None:
        return pd.DataFrame(columns=DST_COLUMNS)
    df = df.rename(columns={time_col: "time_tag", dst_col: "dst"})
    if "source_type" not in df.columns:
        df["source_type"] = "realtime"
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df.dropna(subset=["time_tag"])
    df = df[(df["time_tag"] >= start_dt) & (df["time_tag"] <= end_dt)]
    return df[["time_tag", "dst", "source_type"]]


def _fill_with_html(
    df: pd.DataFrame, html_df: pd.DataFrame, prefer_html: bool = False
) -> pd.DataFrame:
    if html_df.empty:
        return df
    left = df.set_index("time_tag")
    html = html_df.set_index("time_tag")
    if prefer_html:
        replacements = html["dst"].dropna()
        if not replacements.empty:
            left.loc[replacements.index, "dst"] = replacements
    else:
        missing_idx = left.index[left["dst"].isna()]
        if not missing_idx.empty:
            replacements = html.reindex(missing_idx)["dst"]
            left.loc[missing_idx, "dst"] = replacements
    return left.reset_index()
