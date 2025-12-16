from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, List, Optional

import pandas as pd
import requests

from common.http import http_get

REALTIME_WINDOW_DAYS = 7
REALTIME_URL = "https://www.sidc.be/cactus/out/cmecat.txt"
CATALOG_BASE = "https://www.sidc.be/cactus/catalog/LASCO/2_5_0/"

OUTPUT_COLUMNS = [
    "event_id",
    "catalog_month",
    "cme_number",
    "time_tag",
    "dt_minutes",
    "position_angle",
    "angular_width",
    "median_velocity",
    "velocity_variation",
    "min_velocity",
    "max_velocity",
    "halo_class",
]


def download_cme_catalog(
    start_date: date,
    end_date: date,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download CACTUS CME catalogue rows covering the requested date range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    frames: List[pd.DataFrame] = []
    for target in _iter_catalog_targets(start_date, end_date):
        url = build_catalog_url(target)
        text = _download_catalog(url, session=session)
        df = parse_cme_table(text)
        if df.empty:
            continue
        df["catalog_month"] = date(target.year, target.month, 1)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    prepared = prepare_cme_dataframe(combined, start_date, end_date)
    return prepared


def build_catalog_url(target_date: date) -> str:
    """
    Construct the catalogue URL for the specified date.
    """
    today = date.today()
    if target_date <= today and (today - target_date) <= timedelta(days=REALTIME_WINDOW_DAYS):
        return REALTIME_URL

    base = CATALOG_BASE
    if (target_date.year, target_date.month) >= (2010, 8):
        base += "qkl/"
    return f"{base}{target_date:%Y}/{target_date:%m}/cmecat.txt"


def parse_cme_table(raw_text: str) -> pd.DataFrame:
    """
    Parse the CME table from the CACTUS catalogue dump.
    """
    lines = raw_text.splitlines()
    collecting = False
    header_line = None
    data_rows: List[List[str]] = []

    for line in lines:
        if line.startswith("# Flow"):
            break

        if not collecting and line.startswith("# CME"):
            header_line = line
            collecting = True
            continue

        if not collecting:
            continue

        if not line.strip() or line.startswith("#"):
            continue

        parts = [segment.strip() for segment in line.split("|")]
        data_rows.append(parts)

    if header_line is None:
        return pd.DataFrame()

    columns = _extract_columns(header_line)
    normalized_rows = [
        row[: len(columns)] + [""] * max(0, len(columns) - len(row)) for row in data_rows
    ]
    return pd.DataFrame(normalized_rows, columns=columns)


def prepare_cme_dataframe(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Return dataframe with parsed timestamps, numeric columns, and range filtering applied.
    """
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prepared = df.copy()
    prepared = prepared.rename(
        columns={
            "CME": "cme_number",
            "t0": "time_tag",
            "dt0": "dt_minutes",
            "pa": "position_angle",
            "da": "angular_width",
            "v": "median_velocity",
            "dv": "velocity_variation",
            "minv": "min_velocity",
            "maxv": "max_velocity",
            "halo?": "halo_class",
        }
    )
    prepared["time_tag"] = pd.to_datetime(prepared["time_tag"], format="%Y/%m/%d %H:%M", errors="coerce")
    numeric_columns = [
        "dt_minutes",
        "position_angle",
        "angular_width",
        "median_velocity",
        "velocity_variation",
        "min_velocity",
        "max_velocity",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["cme_number"] = prepared["cme_number"].astype(str).str.strip()
    prepared["halo_class"] = prepared.get("halo_class", "").astype(str).str.strip()
    prepared.loc[prepared["halo_class"] == "", "halo_class"] = None
    prepared["catalog_month"] = pd.to_datetime(prepared["catalog_month"], errors="coerce").dt.date

    prepared = prepared.dropna(subset=["time_tag"])
    mask = (prepared["time_tag"].dt.date >= start_date) & (prepared["time_tag"].dt.date <= end_date)
    prepared = prepared.loc[mask]

    if prepared.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prepared["event_id"] = prepared.apply(_build_event_id, axis=1)
    prepared = prepared.drop_duplicates(subset=["event_id"])
    prepared = prepared.sort_values("time_tag")

    return prepared.reindex(columns=OUTPUT_COLUMNS)


def _build_event_id(row) -> str:
    timestamp = row.get("time_tag")
    if pd.isna(timestamp):
        timestamp = None
    if timestamp is not None and hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    cme_number = str(row.get("cme_number") or "").zfill(4)
    if timestamp is None:
        return cme_number
    return f"{timestamp:%Y%m%d%H%M}-{cme_number}"


def _download_catalog(url: str, session: Optional[requests.Session] = None) -> str:
    response = http_get(url, session=session, log_name="CME", timeout=60)
    if response is None:
        raise RuntimeError(f"Failed to download catalogue from {url}")
    return response.text


def _extract_columns(header_line: str) -> List[str]:
    header = header_line.lstrip("#").strip()
    return [segment.strip() for segment in header.split("|") if segment.strip()]


def _iter_catalog_targets(start_date: date, end_date: date) -> Iterable[date]:
    current = date(start_date.year, start_date.month, 1)
    final = date(end_date.year, end_date.month, 1)

    while current <= final:
        yield current
        year = current.year + (current.month // 12)
        month = (current.month % 12) + 1
        current = date(year, month, 1)

    today = date.today()
    recent_cutoff = today - timedelta(days=REALTIME_WINDOW_DAYS)
    if start_date <= today and end_date >= recent_cutoff:
        yield today
