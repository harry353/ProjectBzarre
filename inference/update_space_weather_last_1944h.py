from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from database_builder.discovery import load_data_source_classes
from database_builder.helpers import build_source_kwargs, friendly_name
from database_builder.logging_utils import stamp
from database_builder.windows import iter_date_windows
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.sunspot_number.sunspot_number_data_source import (
    SunspotNumberDataSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DB = PROJECT_ROOT / "inference" / "space_weather_last_1944h.db"
HOURS_BACK = 1944
TARGET_CLASS_NAMES = {
    "IMFACEDataSource",
    "SolarWindACEDataSource",
    "IMFDSCOVRDataSource",
    "SolarWindDSCOVRDataSource",
    "DstDataSource",
    "KpIndexDataSource",
    "CMEDataSource",
    "RadioFluxDataSource",
}
TIME_COLUMNS = {
    "ace_mfi": "time_tag",
    "ace_swepam": "time_tag",
    "dscovr_f1m": "time_tag",
    "dscovr_m1m": "time_tag",
    "dst_index": "time_tag",
    "kp_index": "time_tag",
    "lasco_cme_catalog": "time_tag",
    "radio_flux": "time_tag",
    "sunspot_numbers": "time_tag",
}


def main() -> None:
    run_started = time.time()
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=HOURS_BACK)
    start_date = start_dt.date()
    end_date = end_dt.date()

    print(
        stamp(
            "Refreshing daily inference snapshot from live sources "
            f"({start_date.isoformat()} -> {end_date.isoformat()})..."
        )
    )

    _clear_forward_fills()
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    warehouse = SpaceWeatherWarehouse(str(OUTPUT_DB))

    classes = [
        cls
        for cls in load_data_source_classes()
        if cls.__name__ in TARGET_CLASS_NAMES
    ]

    _refresh_sunspot(start_date, end_date, warehouse)

    ingest_lock = threading.Lock()

    def _run_source(cls) -> None:
        label = friendly_name(cls.__name__)
        for window_start, window_end in iter_date_windows(start_date, end_date):
            print(stamp(f"[INFO] Processing {label} for {window_start} -> {window_end}"))
            try:
                source = cls(**build_source_kwargs(cls, window_start, window_end))
                df = source.download()
            except Exception:
                continue
            if df is None or getattr(df, "empty", False):
                continue
            try:
                with ingest_lock:
                    source.ingest(df, warehouse=warehouse)
            except Exception:
                continue

    with ThreadPoolExecutor(max_workers=min(len(classes), 6)) as executor:
        futures = [executor.submit(_run_source, cls) for cls in classes]
        for f in as_completed(futures):
            f.result()

    _crop_to_window(start_dt, end_dt)
    _enforce_sunspot_utc()
    _enforce_kp_utc()
    _enforce_radio_flux_utc()
    _extend_sunspot_to_now()
    _extend_kp_to_now()
    _extend_radio_flux_to_now()

    duration = time.time() - run_started
    print(stamp(f"Update complete in {duration:.2f} seconds"))


def _clear_forward_fills() -> None:
    if not OUTPUT_DB.exists():
        return
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.execute("DELETE FROM kp_index WHERE source_type = 'forward_fill'")
        conn.execute("DELETE FROM sunspot_numbers WHERE source_type = 'forward_fill'")
        conn.execute("DELETE FROM radio_flux WHERE source_type = 'forward_fill'")
        conn.commit()


def _crop_to_window(start_dt: datetime, end_dt: datetime) -> None:
    start_cutoff = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_cutoff = end_dt.strftime("%Y-%m-%d %H:%M:%S")
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    with sqlite3.connect(OUTPUT_DB) as conn:
        for table, column in TIME_COLUMNS.items():
            if table == "sunspot_numbers":
                conn.execute(
                    f"DELETE FROM {table} WHERE {column} IS NULL "
                    f"OR date({column}) < date(?) "
                    f"OR date({column}) > date(?)",
                    (start_date, end_date),
                )
            else:
                conn.execute(
                    f"DELETE FROM {table} WHERE {column} IS NULL "
                    f"OR datetime({column}) < datetime(?) "
                    f"OR datetime({column}) > datetime(?)",
                    (start_cutoff, end_cutoff),
                )
        conn.commit()


def _extend_kp_to_now() -> None:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    with sqlite3.connect(OUTPUT_DB) as conn:
        row = conn.execute(
            "SELECT time_tag, kp_index FROM kp_index "
            "ORDER BY datetime(time_tag) DESC LIMIT 1"
        ).fetchone()
        if not row:
            return
        last_dt = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(last_dt):
            return
        last_value = row[1]
        current = last_dt + timedelta(hours=1)
        while current <= now:
            conn.execute(
                "INSERT OR REPLACE INTO kp_index "
                "(time_tag, kp_index, source_type) VALUES (?, ?, ?)",
                (
                    current.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    float(last_value),
                    "forward_fill",
                ),
            )
            current += timedelta(hours=1)
        conn.commit()


def _extend_radio_flux_to_now() -> None:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    with sqlite3.connect(OUTPUT_DB) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(radio_flux)")}
        if "source_type" not in cols:
            conn.execute("ALTER TABLE radio_flux ADD COLUMN source_type TEXT")
        row = conn.execute(
            "SELECT time_tag, observed_flux, adjusted_flux, ursi_flux FROM radio_flux "
            "ORDER BY datetime(time_tag) DESC LIMIT 1"
        ).fetchone()
        if not row:
            return
        last_dt = pd.to_datetime(row[0], errors="coerce", utc=True)
        if pd.isna(last_dt):
            return

        last_obs, last_adj, last_ursi = row[1], row[2], row[3]
        current = last_dt.to_pydatetime() + timedelta(hours=1)
        while current <= now:
            conn.execute(
                "INSERT OR REPLACE INTO radio_flux "
                "(time_tag, observed_flux, adjusted_flux, ursi_flux, source_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    current.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    None if last_obs is None else float(last_obs),
                    None if last_adj is None else float(last_adj),
                    None if last_ursi is None else float(last_ursi),
                    "forward_fill",
                ),
            )
            current += timedelta(hours=1)
        conn.commit()


def _extend_sunspot_to_now() -> None:
    target_date = datetime.now(timezone.utc).date() + timedelta(days=1)
    with sqlite3.connect(OUTPUT_DB) as conn:
        row = conn.execute(
            "SELECT time_tag, sunspot_number FROM sunspot_numbers "
            "ORDER BY date(time_tag) DESC LIMIT 1"
        ).fetchone()
        if not row:
            return
        last_dt = pd.to_datetime(row[0], errors="coerce")
        if pd.isna(last_dt):
            return
        last_value = row[1]
        current = last_dt.date() + timedelta(days=1)
        while current <= target_date:
            conn.execute(
                "INSERT OR REPLACE INTO sunspot_numbers "
                "(time_tag, sunspot_number, source_type) VALUES (?, ?, ?)",
                (
                    f"{current.isoformat()} 00:00:00+00:00",
                    float(last_value),
                    "forward_fill",
                ),
            )
            current += timedelta(days=1)
        conn.commit()


def _refresh_sunspot(start_date, end_date, warehouse) -> None:
    try:
        source = SunspotNumberDataSource(days=(start_date, end_date))
        df = source.download()
    except Exception:
        return
    if df is None or df.empty:
        return
    df = df.copy()
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True).dt.strftime(
        "%Y-%m-%d 00:00:00+00:00"
    )
    df = df.dropna(subset=["time_tag"])
    try:
        source.ingest(df, warehouse=warehouse)
    except Exception:
        return


def _enforce_sunspot_utc() -> None:
    if not OUTPUT_DB.exists():
        return
    with sqlite3.connect(OUTPUT_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag, sunspot_number, source_type FROM sunspot_numbers",
            conn,
        )
        if df.empty or "time_tag" not in df.columns:
            return
        parsed = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df["time_tag"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        df = df.dropna(subset=["time_tag"])
        conn.execute("DELETE FROM sunspot_numbers")
        conn.executemany(
            "INSERT OR REPLACE INTO sunspot_numbers "
            "(time_tag, sunspot_number, source_type) VALUES (?, ?, ?)",
            df[["time_tag", "sunspot_number", "source_type"]]
            .where(df.notna(), None)
            .itertuples(index=False, name=None),
        )
        conn.commit()


def _enforce_kp_utc() -> None:
    if not OUTPUT_DB.exists():
        return
    with sqlite3.connect(OUTPUT_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag, kp_index, source_type FROM kp_index",
            conn,
        )
        if df.empty or "time_tag" not in df.columns:
            return
        parsed = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df["time_tag"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        df = df.dropna(subset=["time_tag"])
        conn.execute("DELETE FROM kp_index")
        conn.executemany(
            "INSERT OR REPLACE INTO kp_index "
            "(time_tag, kp_index, source_type) VALUES (?, ?, ?)",
            df[["time_tag", "kp_index", "source_type"]]
            .where(df.notna(), None)
            .itertuples(index=False, name=None),
        )
        conn.commit()


def _enforce_radio_flux_utc() -> None:
    if not OUTPUT_DB.exists():
        return
    with sqlite3.connect(OUTPUT_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag, observed_flux, adjusted_flux, ursi_flux, source_type FROM radio_flux",
            conn,
        )
        if df.empty or "time_tag" not in df.columns:
            return
        parsed = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df["time_tag"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
        df = df.dropna(subset=["time_tag"])
        conn.execute("DELETE FROM radio_flux")
        conn.executemany(
            "INSERT OR REPLACE INTO radio_flux "
            "(time_tag, observed_flux, adjusted_flux, ursi_flux, source_type) "
            "VALUES (?, ?, ?, ?, ?)",
            df[
                [
                    "time_tag",
                    "observed_flux",
                    "adjusted_flux",
                    "ursi_flux",
                    "source_type",
                ]
            ]
            .where(df.notna(), None)
            .itertuples(index=False, name=None),
        )
        conn.commit()


if __name__ == "__main__":
    main()
