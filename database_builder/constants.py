from __future__ import annotations

import os
import sys
import threading
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_DB_OVERRIDE = os.environ.get("SPACE_WEATHER_DB_PATH")
if _DB_OVERRIDE:
    DB_PATH = Path(_DB_OVERRIDE).expanduser()
else:
    DB_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
DATA_SOURCES_DIR = PROJECT_ROOT / "data_sources"
STATUS_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "data_source_status.csv"
MODULE_SUFFIX = "_data_source"
DEFAULT_START = date(1998, 1, 1)
DEFAULT_END = None
CHUNK_DAYS = 30
BUILD_FROM_REALTIME = True
REALTIME_BACKFILL_DAYS = 3
TRACKER_TIME_COLUMNS = {
    "CMEDataSource": ["Datetime", "time21_5"],
}
TRACKER_LOCK = threading.Lock()
LOG_TIME_FORMAT = "%H:%M:%S"

__all__ = [
    "PROJECT_ROOT",
    "DB_PATH",
    "DATA_SOURCES_DIR",
    "STATUS_PATH",
    "MODULE_SUFFIX",
    "DEFAULT_START",
    "DEFAULT_END",
    "CHUNK_DAYS",
    "BUILD_FROM_REALTIME",
    "REALTIME_BACKFILL_DAYS",
    "TRACKER_TIME_COLUMNS",
    "TRACKER_LOCK",
    "LOG_TIME_FORMAT",
]
