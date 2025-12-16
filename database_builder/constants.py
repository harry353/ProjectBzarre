from __future__ import annotations

import os
import sys
import threading
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
DATA_SOURCES_DIR = PROJECT_ROOT / "data_sources"
STATUS_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "data_source_status.csv"
MODULE_SUFFIX = "_data_source"
DEFAULT_START = date(1998, 1, 1)
DEFAULT_END = date(2025, 11, 30)
CHUNK_DAYS = 30
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
    "TRACKER_TIME_COLUMNS",
    "TRACKER_LOCK",
    "LOG_TIME_FORMAT",
]
