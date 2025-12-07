from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import os
import pkgutil
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Type
from concurrent.futures import ThreadPoolExecutor
import threading

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from space_weather_api import SpaceWeatherAPI  # noqa: E402
from space_weather_warehouse import SpaceWeatherWarehouse  # noqa: E402
from common.http import set_thread_session

DB_PATH = PROJECT_ROOT / "space_weather.db"
DATA_SOURCES_DIR = PROJECT_ROOT / "data_sources"
MODULE_SUFFIX = "_data_source"
DEFAULT_START = date(2005, 1, 1)
DEFAULT_END = date(2005, 1, 7)
STATUS_PATH = PROJECT_ROOT / "data_source_status.csv"
TRACKER_TIME_COLUMNS = {
    "CMEDataSource": ["Datetime", "time21_5"],
}
TRACKER_LOCK = threading.Lock()


def parse_args() -> Tuple[date, date]:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the warehouse database by iterating every data source "
            "one week at a time."
        )
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=DEFAULT_START,
        help="Start date in YYYY-MM-DD (default: 2000-01-01)",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=DEFAULT_END,
        help="End date in YYYY-MM-DD (default: today)",
    )

    args = parser.parse_args()
    end = args.end or date.today()
    if args.start > end:
        parser.error("start date cannot be after end date")
    return args.start, end


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def reset_database(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()


def iter_data_source_modules() -> Iterator[str]:
    prefix = "data_sources."
    for module in pkgutil.walk_packages([str(DATA_SOURCES_DIR)], prefix=prefix):
        if module.ispkg:
            continue
        if module.name.split(".")[-1].endswith(MODULE_SUFFIX):
            yield module.name


def load_data_source_classes() -> List[Type[SpaceWeatherAPI]]:
    classes: List[Type[SpaceWeatherAPI]] = []
    for module_name in iter_data_source_modules():
        module = importlib.import_module(module_name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, SpaceWeatherAPI) and obj is not SpaceWeatherAPI:
                classes.append(obj)
    classes.sort(key=lambda cls: cls.__name__)
    return classes


def iter_week_ranges(start_date: date, end_date: date) -> Iterator[Tuple[date, date]]:
    cursor = start_date
    delta = timedelta(days=7)
    while cursor <= end_date:
        stop = min(cursor + timedelta(days=6), end_date)
        yield cursor, stop
        cursor += delta


def build_source_kwargs(
    cls: Type[SpaceWeatherAPI],
    start_date: date,
    end_date: date,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"days": (start_date, end_date)}
    if cls.__name__ == "SuperMAGDataSource":
        kwargs["logon"] = os.environ.get("SUPERMAG_LOGON", "haris262")
    return kwargs


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
            stamp = (row.get("latest_timestamp") or "").strip()
            tracker[source] = _parse_timestamp(stamp)

    # Persist again in case new sources were added since the last run.
    _write_status_file(path, class_names, tracker)
    return tracker, False


def _write_status_file(path: Path, class_names: List[str], tracker: Dict[str, datetime | None]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "latest_timestamp"])
        for name in class_names:
            value = tracker.get(name)
            stamp = value.isoformat(timespec="seconds") if value else ""
            writer.writerow([name, stamp])


def process_source_week(
    cls: Type[SpaceWeatherAPI],
    warehouse: SpaceWeatherWarehouse,
    week_start: date,
    week_end: date,
    tracker: Dict[str, datetime | None],
    class_names: List[str],
    lock: threading.Lock,
) -> None:
    class_name = cls.__name__
    friendly = _friendly_name(class_name)
    boundary = tracker.get(class_name)
    if boundary is not None and boundary.date() >= week_end:
        print(f"[SKIP] {friendly} already processed through {boundary.date().isoformat()}")
        return

    window_label = f"{week_start.isoformat()} -> {week_end.isoformat()}"
    print(f"Processing {friendly} for {window_label}...")

    try:
        source = cls(**build_source_kwargs(cls, week_start, week_end))
        df = source.download()
    except Exception as exc:
        print(f"[ERROR] {friendly} download failed for {window_label}: {exc}")
        return

    if df is None or df.empty:
        print(f"[INFO] No data for {friendly} during {window_label}. Continuing.")
        return

    try:
        inserted = source.ingest(df, warehouse=warehouse)
    except Exception as exc:
        print(f"[ERROR] {friendly} ingest failed for {window_label}: {exc}")
        return

    if inserted:
        print(f"[OK] Inserted {inserted} rows for {window_label}.")
        _record_latest_timestamp(class_name, df, tracker, class_names, lock)
    else:
        print(f"[INFO] {friendly} returned data but nothing was stored for {window_label}.")


def _record_latest_timestamp(
    class_name: str,
    df,
    tracker: Dict[str, datetime | None],
    class_names: List[str],
    lock: threading.Lock,
) -> None:
    latest = _determine_latest_timestamp(df, class_name)
    if latest is None:
        return

    with lock:
        previous = tracker.get(class_name)
        if previous is None or latest > previous:
            tracker[class_name] = latest
            _write_status_file(STATUS_PATH, class_names, tracker)


def _determine_latest_timestamp(df, class_name: str | None = None) -> datetime | None:
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


def main() -> None:
    start_date, end_date = parse_args()
    run_started = time.time()
    print(
        f"Processing range {start_date.isoformat()} -> {end_date.isoformat()}"
    )

    classes = load_data_source_classes()
    if not classes:
        print("No data sources found. Nothing to do.")
        return

    class_names = [cls.__name__ for cls in classes]
    tracker, is_new_tracker = load_or_initialize_tracker(STATUS_PATH, class_names)

    if is_new_tracker or not DB_PATH.exists():
        print("Starting with a fresh database.")
        reset_database(DB_PATH)
    else:
        print(
            f"Resuming from existing status file ({STATUS_PATH}) and database ({DB_PATH})."
        )

    warehouse = SpaceWeatherWarehouse(str(DB_PATH))
    session_map = {cls.__name__: requests.Session() for cls in classes}
    last_request = {cls.__name__: 0.0 for cls in classes}

    week_ranges = list(iter_week_ranges(start_date, end_date))
    total_weeks = len(week_ranges)
    for index, (week_start, week_end) in enumerate(week_ranges, start=1):
        print(
            f"\n=== Week {index} / {total_weeks}: {week_start.isoformat()} -> {week_end.isoformat()} ==="
        )
        with ThreadPoolExecutor(max_workers=len(classes) or 1) as executor:
            futures = []
            for cls in classes:
                name = cls.__name__
                elapsed = time.time() - last_request[name]
                if last_request[name] and elapsed < 2:
                    time.sleep(2 - elapsed)
                futures.append(
                    executor.submit(
                        _run_single_week,
                        cls,
                        warehouse,
                        week_start,
                        week_end,
                        tracker,
                        class_names,
                        TRACKER_LOCK,
                        session_map[name],
                    )
                )
                last_request[name] = time.time()
            for future in futures:
                future.result()

    duration = time.time() - run_started
    minutes = duration / 60
    print(
        "\nDatabase regeneration complete in "
        f"{duration:.2f} seconds ({minutes:.2f} minutes)."
    )
    for session in session_map.values():
        session.close()


def _friendly_name(class_name: str) -> str:
    base = re.sub(r"DataSource$", "", class_name)
    tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[0-9]+", base)
    label = " ".join(tokens) if tokens else base
    return f"{label} data"


def _run_single_week(
    cls: Type[SpaceWeatherAPI],
    warehouse: SpaceWeatherWarehouse,
    week_start: date,
    week_end: date,
    tracker: Dict[str, datetime | None],
    class_names: List[str],
    lock: threading.Lock,
    session: requests.Session,
) -> None:
    set_thread_session(session)
    process_source_week(cls, warehouse, week_start, week_end, tracker, class_names, lock)


if __name__ == "__main__":
    main()
