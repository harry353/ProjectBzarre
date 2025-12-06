from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from common.http import http_get

LASCO_BASE_URL = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/"
FILENAME_TEMPLATE = "univ{year}_{month:02d}.txt"


def download_cme_lasco(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Download LASCO CME entries from the CDAW archive for the given range.
    """
    if start_date > end_date:
        return pd.DataFrame()

    monthly_cache: Dict[Tuple[int, int], Optional[str]] = {}
    rows: List[Dict[str, object]] = []

    for single_date in _iter_days(start_date, end_date):
        text = _get_month_text(single_date.year, single_date.month, monthly_cache)
        if not text:
            continue
        rows.extend(_parse_day_entries(text, single_date))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], errors="coerce", utc=True
    )
    df["event_key"] = df.apply(_build_event_key, axis=1)
    df = df.drop_duplicates(subset=["event_key"])
    return df.sort_values("Datetime", na_position="last").reset_index(drop=True)


def _fetch_month_text(year: int, month: int) -> Optional[str]:
    filename = FILENAME_TEMPLATE.format(year=year, month=month)
    url = f"{LASCO_BASE_URL}{filename}"
    response = http_get(url, timeout=60, log_name="LASCO CME", raise_for_status=False)
    if response is None:
        return None
    if response.status_code == 404:
        # No entries for that month.
        return None
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def _parse_day_entries(text: str, target_date: date) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    target_prefix = target_date.strftime("%Y/%m/%d")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith(target_prefix):
            continue

        parts = line.split()
        parsed = _parse_row(parts)
        if parsed is not None:
            rows.append(parsed)

    return rows


def _get_month_text(year: int, month: int, cache: Dict[Tuple[int, int], Optional[str]]) -> Optional[str]:
    key = (year, month)
    if key not in cache:
        cache[key] = _fetch_month_text(year, month)
    return cache[key]


def _iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _parse_row(parts: List[str]) -> Optional[Dict[str, object]]:
    if len(parts) < 3:
        return None

    date_val = parts[0]
    time_val = parts[1]

    cpa_raw = parts[2]
    if cpa_raw == "Halo":
        cpa = None
        width = 360.0
        offset = 1
    else:
        cpa = _parse_numeric(cpa_raw)
        width = _parse_numeric(_get(parts, 3))
        offset = 0

    lin_speed = _parse_numeric(_get(parts, 3 + offset))
    init_speed = _parse_numeric(_get(parts, 4 + offset))
    final_speed = _parse_numeric(_get(parts, 5 + offset))
    speed_20r = _parse_numeric(_get(parts, 6 + offset))
    accel = _parse_numeric(_get(parts, 7 + offset))
    mass = _parse_numeric(_get(parts, 8 + offset))
    kinetic = _parse_numeric(_get(parts, 9 + offset))

    mpa_raw = _get(parts, 10 + offset)
    mpa = None if mpa_raw == "Halo" else _parse_numeric(mpa_raw)

    remark_index = 11 + offset
    remark = " ".join(parts[remark_index:]) if len(parts) > remark_index else ""

    return {
        "date": date_val,
        "time": time_val,
        "CPA": cpa,
        "Width": width,
        "Linear_Speed": lin_speed,
        "Initial_Speed": init_speed,
        "Final_Speed": final_speed,
        "Speed_20R": speed_20r,
        "Acceleration": accel,
        "Mass": mass,
        "Kinetic_Energy": kinetic,
        "MPA": mpa,
        "Remarks": remark,
    }


def _get(parts: List[str], index: int) -> Optional[str]:
    if index is None or index >= len(parts) or index < 0:
        return None
    return parts[index]


def _parse_numeric(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-------":
        return None

    value = value.replace("*", "")

    try:
        return float(value)
    except ValueError:
        return None


def _build_event_key(row: pd.Series) -> str:
    remarks = row.get("Remarks") or ""
    safe_remarks = "_".join(str(remarks).split())
    return f"{row['date']}_{row['time']}_{safe_remarks}"
