from datetime import date, datetime, timedelta, timezone
from typing import Iterator, Optional

import pandas as pd
import requests
import re

from common.http import http_get
from database_builder.constants import BUILD_FROM_REALTIME, REALTIME_BACKFILL_DAYS

# Three Kyoto WDC base URLs covering different data maturity levels.
DST_FINAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_final"
DST_PROVISIONAL_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_provisional"
DST_REALTIME_BASE = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime"
# Column names that every returned DataFrame is guaranteed to contain.
DST_COLUMNS = ["time_tag", "dst", "source_type"]
# NOAA SWPC JSON feed for the most recent Kyoto-derived Dst values.
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
        # Pre-2005 data and realtime months only have the HTML index page.
        if month_start <= date(2004, 12, 31) or base_url == DST_REALTIME_BASE:
            df = _fetch_month_html(base_url, month_start, session)
        else:
            # Attempt the machine-readable .for.request text file first.
            url = _build_month_url(base_url, month_start)
            df = _fetch_month(url, month_start, session)
            html_df = _fetch_month_html(base_url, month_start, session)
            if df is None:
                df = html_df
            elif html_df is not None:
                # Use the HTML page to fill any gaps in the text-file parse.
                df = _fill_with_html(df, html_df, prefer_html=base_url == DST_REALTIME_BASE)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=DST_COLUMNS)

    data = pd.concat(frames).sort_values("time_tag").reset_index(drop=True)
    # Kyoto timestamps are end-of-hour; shift +1h to convert to start-of-hour convention.
    data["time_tag"] = data["time_tag"] + pd.Timedelta(hours=1)
    mask = (data["time_tag"].dt.date >= start_date) & (
        data["time_tag"].dt.date <= end_date
    )
    data = data.loc[mask].reset_index(drop=True)
    data["source_type"] = "archive"
    # Keep the row with a non-null Dst value when duplicates exist for the same timestamp.
    data["_valid"] = data["dst"].notna().astype(int)
    data = (
        data.sort_values(["time_tag", "_valid"])
        .drop_duplicates(subset="time_tag", keep="last")
        .drop(columns="_valid")
        .reset_index(drop=True)
    )
    if BUILD_FROM_REALTIME and REALTIME_BACKFILL_DAYS > 0:
        # Overwrite the tail of the archive with fresher SWPC realtime readings.
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
    # Merge the latest SWPC point (rolled +1h) into the archive data
    data = _merge_latest_swpc(data, session)
    data = data.dropna(subset=["dst"]).reset_index(drop=True)
    data = data.reindex(columns=DST_COLUMNS)
    return data


def _month_range(start: date, end: date) -> Iterator[date]:
    # Yield the first day of each calendar month from start through end inclusive.
    current = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while current <= last:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _base_for_month(month_date: date) -> str:
    # Select the appropriate Kyoto WDC base URL based on the data maturity for the month.
    if month_date < date(2020, 12, 31):
        return DST_FINAL_BASE
    if month_date <= date(2025, 6, 30):
        return DST_PROVISIONAL_BASE
    return DST_REALTIME_BASE


def _build_month_url(base_url: str, month_date: date) -> str:
    # Construct the URL for the machine-readable .for.request text file for a given month.
    year = month_date.year
    suffix = f"{str(year)[2:]}{month_date.month:02d}.for.request"
    # The present month on the realtime server lives under a fixed "presentmonth" path.
    if base_url == DST_REALTIME_BASE and month_date == date.today().replace(day=1):
        return f"{base_url}/presentmonth/dst{suffix}"
    return f"{base_url}/{year}{month_date.month:02d}/dst{suffix}"


def _build_month_html_url(base_url: str, month_date: date) -> str:
    # Construct the URL for the HTML index page, which is used as a fallback data source.
    if base_url == DST_REALTIME_BASE and month_date == date.today().replace(day=1):
        return f"{base_url}/presentmonth/index.html"
    return f"{base_url}/{month_date.year}{month_date.month:02d}/index.html"


def _fetch_month(url: str, month_start: date, session: requests.Session):
    # Download and parse the Kyoto .for.request fixed-format text file for one month.
    resp = http_get(url, session=session, log_name="Dst", timeout=60)
    if resp is None:
        return None

    rows = []
    for line in resp.text.splitlines():
        # Each data line starts with "DST"; skip headers and blank lines.
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
    # Parse hourly Dst values from the HTML <pre> block on the Kyoto index page.
    url = _build_month_html_url(base_url, month_start)
    resp = http_get(url, session=session, log_name="DstHTML", timeout=60)
    if resp is None:
        return None

    capture = False
    rows = []
    for line in resp.text.splitlines():
        # Data is wrapped in a <pre> block — start capturing on the opening tag.
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
        # Pad to 24 hours with None if the line contains fewer readings.
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
    # Extract the day-of-month and 24 hourly value tokens from a .for.request data line.
    try:
        day = int(line.split("*")[1][:2])
    except Exception:
        return None

    # Insert spaces before negative signs that are glued to the preceding value.
    parts = _fix_glued_negatives(line).split()
    if len(parts) < 3:
        return None

    # The first two tokens are metadata; hourly readings start at index 2.
    return day, parts[2:]


def _fix_glued_negatives(line: str) -> str:
    # Insert a space before any '-' that is not already preceded by a space or another '-'.
    fixed = []
    prev = " "
    for ch in line:
        if ch == "-" and prev not in [" ", "-"]:
            fixed.append(" ")
        fixed.append(ch)
        prev = ch
    return "".join(fixed)


def _parse_dst_value(token: str):
    # Convert a token to int, returning None for non-numeric or out-of-range values.
    try:
        value = int(token)
    except Exception:
        return None
    # Values outside ±2000 nT are treated as fill/missing indicators.
    if -2000 <= value <= 2000:
        return value
    return None


def _fetch_swpc_dst(
    start_dt: datetime, end_dt: datetime, session: requests.Session
) -> pd.DataFrame:
    # Download the SWPC Kyoto-Dst JSON feed and return rows within the requested window.
    resp = http_get(DST_SWPC_URL, session=session, log_name="DstSWPC", timeout=30)
    if resp is None:
        return pd.DataFrame(columns=DST_COLUMNS)
    try:
        payload = resp.json()
    except Exception:
        return pd.DataFrame(columns=DST_COLUMNS)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=DST_COLUMNS)
    # SWPC JSON may use a header-row-first format or a list-of-dicts format.
    if isinstance(payload[0], list):
        header = payload[0]
        rows = payload[1:]
        df = pd.DataFrame(rows, columns=header)
    else:
        df = pd.DataFrame(payload)
    # Normalise column names case-insensitively to "time_tag" and "dst".
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


def _fetch_latest_swpc(session: requests.Session) -> dict | None:
    """
    Fetch the most recent SWPC Dst reading.
    """
    resp = http_get(DST_SWPC_URL, session=session, log_name="DstSWPCLatest", timeout=30)
    if resp is None:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    # Handle both header-row-first and list-of-dicts JSON layouts.
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
        return None
    df = df.rename(columns={time_col: "time_tag", dst_col: "dst"})
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df.dropna(subset=["time_tag", "dst"]).sort_values("time_tag")
    if df.empty:
        return None
    # Return only the single most recent reading as a plain dict.
    latest = df.iloc[-1]
    return {"time_tag": latest["time_tag"], "dst": int(latest["dst"])}


def _merge_latest_swpc(data: pd.DataFrame, session: requests.Session) -> pd.DataFrame:
    """
    Roll the latest SWPC timestamp +1h and replace/append into the archive data.
    """
    latest = _fetch_latest_swpc(session)
    if latest is None:
        return data

    # Apply the same +1h end-of-hour correction used for all Kyoto timestamps.
    rolled_ts = latest["time_tag"] + timedelta(hours=1)
    data = data.copy()
    data["time_tag"] = pd.to_datetime(data["time_tag"], errors="coerce", utc=True)
    mask = data["time_tag"] == rolled_ts
    if mask.any():
        # Update in-place if a row for this timestamp already exists.
        data.loc[mask, "dst"] = latest["dst"]
        data.loc[mask, "source_type"] = "swpc_latest"
    else:
        # Append a new row when the latest reading extends beyond the archive.
        data = pd.concat(
            [
                data,
                pd.DataFrame(
                    [{"time_tag": rolled_ts, "dst": latest["dst"], "source_type": "swpc_latest"}]
                ),
            ],
            ignore_index=True,
        )
    return data.sort_values("time_tag").reset_index(drop=True)


def _fill_with_html(
    df: pd.DataFrame, html_df: pd.DataFrame, prefer_html: bool = False
) -> pd.DataFrame:
    # Merge HTML-parsed values into the text-file DataFrame, filling gaps or overwriting if preferred.
    if html_df.empty:
        return df
    left = df.set_index("time_tag")
    html = html_df.set_index("time_tag")
    if prefer_html:
        # Overwrite all available HTML values (used for realtime months where HTML is authoritative).
        replacements = html["dst"].dropna()
        if not replacements.empty:
            left.loc[replacements.index, "dst"] = replacements
    else:
        # Only fill timestamps where the text-file parse produced a null value.
        missing_idx = left.index[left["dst"].isna()]
        if not missing_idx.empty:
            replacements = html.reindex(missing_idx)["dst"]
            left.loc[missing_idx, "dst"] = replacements
    return left.reset_index()
