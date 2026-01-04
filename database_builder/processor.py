from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Dict, List, Tuple, Type
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import requests

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from .constants import CHUNK_DAYS, TRACKER_LOCK
from .helpers import build_source_kwargs, friendly_name
from .logging_utils import stamp
from .tracker import record_latest_timestamp
from .windows import iter_date_windows
from common.http import set_thread_session


def process_sources(
    classes: List[Type[SpaceWeatherAPI]],
    warehouse: SpaceWeatherWarehouse,
    start_date: date,
    end_date: date,
    tracker,
    class_names: List[str],
) -> None:
    week_ranges = list(iter_date_windows(start_date, end_date))
    total_weeks = len(week_ranges)
    print(
        stamp(
            f"Processing {len(classes)} data sources across {total_weeks} "
            f"{CHUNK_DAYS}-day windows..."
        )
    )
    session_map = {cls.__name__: requests.Session() for cls in classes}
    last_request = {cls.__name__: 0.0 for cls in classes}
    throttle_locks = {cls.__name__: threading.Lock() for cls in classes}

    sequential_classes = [cls for cls in classes if not getattr(cls, "RUN_IN_THREADPOOL", True)]
    parallel_classes = [cls for cls in classes if getattr(cls, "RUN_IN_THREADPOOL", True)]

    with ThreadPoolExecutor(max_workers=len(parallel_classes) or 1) as executor:
        futures = []
        for cls in parallel_classes:
            name = cls.__name__
            futures.append(
                executor.submit(
                    _run_source_over_weeks,
                    cls,
                    warehouse,
                    week_ranges,
                    tracker,
                    class_names,
                    TRACKER_LOCK,
                    session_map[name],
                    last_request,
                    throttle_locks[name],
                )
            )

        for cls in sequential_classes:
            name = cls.__name__
            _run_source_over_weeks(
                cls,
                warehouse,
                week_ranges,
                tracker,
                class_names,
                TRACKER_LOCK,
                session_map[name],
                last_request,
                throttle_locks[name],
            )
        for future in futures:
            future.result()

    for session in session_map.values():
        session.close()


def process_source_week(
    cls: Type[SpaceWeatherAPI],
    warehouse: SpaceWeatherWarehouse,
    week_start: date,
    week_end: date,
    tracker,
    class_names: List[str],
    lock: threading.Lock,
) -> None:
    class_name = cls.__name__
    label = friendly_name(class_name)
    boundary = tracker.get(class_name)
    if boundary is not None and boundary.date() >= week_end:
        print(stamp(f"[SKIP] {label} already processed through {boundary.date().isoformat()}"))
        return

    window_label = f"{week_start.isoformat()} -> {week_end.isoformat()}"
    print(stamp(f"[INFO] Processing {label} for {window_label}..."))

    try:
        source = cls(**build_source_kwargs(cls, week_start, week_end))
        df = source.download()
    except Exception as exc:
        print(stamp(f"[ERROR] {label} download failed for {window_label}: {exc}"))
        return

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        print(stamp(f"[INFO] No data for {label} during {window_label}. Continuing."))
        return

    tracker_payload = source.tracker_payload(df)
    source_breakdown = _format_source_breakdown(df)
    try:
        inserted = source.ingest(df, warehouse=warehouse)
    except Exception as exc:
        print(stamp(f"[ERROR] {label} ingest failed for {window_label}: {exc}"))
        return

    if inserted:
        suffix = f" {source_breakdown}" if source_breakdown else ""
        print(stamp(f"[OK] {label} inserted {inserted} rows for {window_label}.{suffix}"))
        with lock:
            record_latest_timestamp(class_name, tracker_payload, tracker, class_names)
    else:
        print(stamp(f"[INFO] {label} returned data but nothing was stored for {window_label}."))


def _run_source_over_weeks(
    cls: Type[SpaceWeatherAPI],
    warehouse: SpaceWeatherWarehouse,
    week_ranges: List[Tuple[date, date]],
    tracker,
    class_names: List[str],
    lock: threading.Lock,
    session: requests.Session,
    last_request: Dict[str, float],
    throttle_lock: threading.Lock,
) -> None:
    class_name = cls.__name__
    label = friendly_name(class_name)
    boundary = tracker.get(class_name)
    boundary_date = boundary.date() if boundary else None
    if boundary_date is not None:
        end_date = max(window[1] for window in week_ranges)
        resume_start = boundary_date + timedelta(days=1)
        pending = list(iter_date_windows(resume_start, end_date))
        if not pending:
            print(stamp(f"[SKIP] {label} already processed through {boundary_date.isoformat()}"))
            return
    else:
        pending = week_ranges

    for week_start, week_end in pending:
        _run_single_week(
            cls,
            warehouse,
            week_start,
            week_end,
            tracker,
            class_names,
            lock,
            session,
            last_request,
            throttle_lock,
        )


def _run_single_week(
    cls: Type[SpaceWeatherAPI],
    warehouse: SpaceWeatherWarehouse,
    week_start: date,
    week_end: date,
    tracker,
    class_names: List[str],
    lock: threading.Lock,
    session: requests.Session,
    last_request: Dict[str, float],
    throttle_lock: threading.Lock,
) -> None:
    class_name = cls.__name__
    _respect_throttle(class_name, last_request, throttle_lock)
    set_thread_session(session)
    process_source_week(cls, warehouse, week_start, week_end, tracker, class_names, lock)


def _respect_throttle(
    class_name: str,
    last_request: Dict[str, float],
    lock: threading.Lock,
    delay: float = 2.0,
) -> None:
    wait = 0.0
    now = time.time()
    with lock:
        previous = last_request.get(class_name, 0.0)
        if previous:
            elapsed = now - previous
            if elapsed < delay:
                wait = delay - elapsed
    if wait > 0:
        time.sleep(wait)
    with lock:
        last_request[class_name] = time.time()


__all__ = ["process_sources"]


def _format_source_breakdown(df: pd.DataFrame) -> str:
    if "source_type" not in df.columns:
        return ""

    source_col = df["source_type"].fillna("archive")
    counts = source_col.value_counts()
    parts = []
    if "time_tag" in df.columns:
        times = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        working = df.copy()
        working["_source_type"] = source_col
        working["_time_tag"] = times
        ranges = (
            working.dropna(subset=["_time_tag"])
            .groupby("_source_type")["_time_tag"]
            .agg(["min", "max"])
        )
        for source, count in counts.items():
            if source in ranges.index:
                start = ranges.loc[source, "min"]
                end = ranges.loc[source, "max"]
                parts.append(f"{count} {source} rows ({start} -> {end})")
            else:
                parts.append(f"{count} {source} rows")
    else:
        for source, count in counts.items():
            parts.append(f"{count} {source} rows")
    if not parts:
        return ""
    return "Source breakdown: " + "; ".join(parts)
