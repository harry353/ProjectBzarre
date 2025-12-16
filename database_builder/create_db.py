from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from space_weather_warehouse import SpaceWeatherWarehouse

from database_builder.constants import (
    DB_PATH,
    DEFAULT_END,
    DEFAULT_START,
    STATUS_PATH,
)
from database_builder.discovery import load_data_source_classes
from database_builder.helpers import reset_database
from database_builder.logging_utils import stamp
from database_builder.processor import process_sources
from database_builder.tracker import load_or_initialize_tracker


def parse_args() -> Tuple[date, date]:
    parser = argparse.ArgumentParser(
        description="Regenerate the warehouse by iterating every data source."
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=DEFAULT_START,
        help="Start date in YYYY-MM-DD (default: 2005-01-01)",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=DEFAULT_END,
        help="End date in YYYY-MM-DD (default: 2005-12-31)",
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


def main() -> None:
    start_date, end_date = parse_args()
    run_started = time.time()
    print(stamp(f"Processing range {start_date.isoformat()} -> {end_date.isoformat()}"))

    classes = load_data_source_classes()
    if not classes:
        print(stamp("No data sources found. Nothing to do."))
        return

    class_names = [cls.__name__ for cls in classes]
    tracker, is_new_tracker = load_or_initialize_tracker(STATUS_PATH, class_names)

    if is_new_tracker or not DB_PATH.exists():
        print(stamp("Starting with a fresh database."))
        reset_database(DB_PATH)
    else:
        print(
            stamp(
                f"Resuming from existing status file ({STATUS_PATH}) and database ({DB_PATH})."
            )
        )

    warehouse = SpaceWeatherWarehouse(str(DB_PATH))
    process_sources(classes, warehouse, start_date, end_date, tracker, class_names)

    duration = time.time() - run_started
    minutes = duration / 60
    print(
        stamp(
            "Database regeneration complete in "
            f"{duration:.2f} seconds ({minutes:.2f} minutes)."
        )
    )


if __name__ == "__main__":
    main()
