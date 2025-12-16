from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .constants import STATUS_PATH, TRACKER_TIME_COLUMNS
def load_or_initialize_tracker(
    path: Path, class_names: List[str]
) -> Tuple[Dict[str, datetime | None], bool]:
    tracker: Dict[str, datetime | None] = {name: None for name in class_names}
    if not path.exists():
        _write_status_file(path, class_names, tracker)
        return tracker, True

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source = (row.get("source") or "").strip()
            if source not in tracker:
                continue
            stamp_value = (row.get("latest_timestamp") or "").strip()
            tracker[source] = _parse_timestamp(stamp_value)

    _write_status_file(path, class_names, tracker)
    return tracker, False


def _write_status_file(
    path: Path, class_names: List[str], tracker: Dict[str, datetime | None]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "latest_timestamp"])
        for name in class_names:
            value = tracker.get(name)
            stamp_value = value.isoformat(timespec="seconds") if value else ""
            writer.writerow([name, stamp_value])


def record_latest_timestamp(
    class_name: str,
    df,
    tracker: Dict[str, datetime | None],
    class_names: List[str],
) -> None:
    latest = determine_latest_timestamp(df, class_name)
    if latest is None:
        return

    previous = tracker.get(class_name)
    if previous is None or latest > previous:
        tracker[class_name] = latest
        _write_status_file(STATUS_PATH, class_names, tracker)


def determine_latest_timestamp(df, class_name: str | None = None) -> datetime | None:
    if df is None or df.empty:
        return None

    preferred = TRACKER_TIME_COLUMNS.get(class_name or "")
    if preferred:
        candidates = []
        for column in preferred:
            if column not in df.columns:
                continue
            series = _coerce_to_datetime_series(df[column])
            if series is not None and series.notna().any():
                candidates.append(series.max())
        normalized = [_normalize_timestamp(value) for value in candidates]
        normalized = [value for value in normalized if value is not None]
        if normalized:
            return max(normalized)

    candidates = []
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        candidates.append(df.index.max())

    for column in df.columns:
        lowered = column.lower()
        if "time" not in lowered and "date" not in lowered:
            continue
        series = _coerce_to_datetime_series(df[column])
        if series is not None and series.notna().any():
            candidates.append(series.max())

    normalized = [_normalize_timestamp(value) for value in candidates]
    normalized = [value for value in normalized if value is not None]
    if not normalized:
        return None
    return max(normalized)


def _normalize_timestamp(value):
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return None


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _coerce_to_datetime_series(series) -> pd.Series | None:
    try:
        converted = pd.to_datetime(series, errors="coerce", utc=True)
    except Exception:
        return None
    return converted


__all__ = [
    "load_or_initialize_tracker",
    "record_latest_timestamp",
    "determine_latest_timestamp",
]
