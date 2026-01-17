from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Optional

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

CLASS_TABLE_MAP = {
    "IMFACEDataSource": "ace_mfi",
    "SolarWindACEDataSource": "ace_swepam",
    "IMFDSCOVRDataSource": "dscovr_m1m",
    "SolarWindDSCOVRDataSource": "dscovr_f1m",
    "DstDataSource": "dst_index",
    "KpIndexDataSource": "kp_index",
    "CMEDataSource": "lasco_cme_catalog",
    "RadioFluxDataSource": "radio_flux",
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
    if not OUTPUT_DB.exists():
        raise FileNotFoundError(f"Snapshot DB missing: {OUTPUT_DB}")

    _remove_recent_data(days=2)

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
        table = CLASS_TABLE_MAP.get(cls.__name__)
        last_ts = _last_table_timestamp(table)
        src_start_date = start_date
        if last_ts is not None:
            src_start_date = max(start_date, (last_ts - timedelta(days=2)).date())

        for w_start, w_end in iter_date_windows(src_start_date, end_date):
            print(stamp(f"[INFO] Processing {label} for {w_start} -> {w_end}"))
            try:
                source = cls(**build_source_kwargs(cls, w_start, w_end))
                df = source.download()
            except Exception as exc:
                print(stamp(f"[ERROR] {label} download failed: {exc}"))
                continue
            if df is None or df.empty:
                continue
            try:
                with ingest_lock:
                    source.ingest(df, warehouse=warehouse)
            except Exception as exc:
                print(stamp(f"[ERROR] {label} ingest failed: {exc}"))

    with ThreadPoolExecutor(max_workers=min(len(classes), 6)) as executor:
        futures = [executor.submit(_run_source, cls) for cls in classes]
        for f in as_completed(futures):
            f.result()

    _normalize_dscovr_inplace()
    _normalize_sunspot_dedup()
    _normalize_kp_dedup()
    _normalize_radio_flux_dedup()

    _extend_sunspot_to_now()
    _extend_kp_to_now()
    _extend_radio_flux_to_now()

    duration = time.time() - run_started
    print(stamp(f"Update complete in {duration:.2f} seconds"))


def _clear_forward_fills() -> None:
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.execute("DELETE FROM kp_index WHERE source_type = 'forward_fill'")
        conn.execute("DELETE FROM sunspot_numbers WHERE source_type = 'forward_fill'")
        conn.execute("DELETE FROM radio_flux WHERE source_type = 'forward_fill'")
        conn.commit()


def _remove_recent_data(days: int) -> None:
    cutoff = (
        datetime.now(timezone.utc)
        .replace(hour=12, minute=0, second=0, microsecond=0)
        - timedelta(days=days)
    ).strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(OUTPUT_DB) as conn:
        for table, col in TIME_COLUMNS.items():
            conn.execute(
                f"DELETE FROM {table} WHERE {col} IS NOT NULL "
                f"AND datetime({col}) >= datetime(?)",
                (cutoff,),
            )
        conn.commit()


def _normalize_dscovr_inplace() -> None:
    with sqlite3.connect(OUTPUT_DB) as conn:
        for table in ("dscovr_f1m", "dscovr_m1m"):
            conn.execute(
                f"""
                UPDATE {table}
                SET time_tag = strftime('%Y-%m-%d %H:%M:%S+00:00', datetime(time_tag))
                WHERE time_tag IS NOT NULL
                """
            )
        conn.commit()


def _normalize_radio_flux_dedup() -> None:
    with sqlite3.connect(OUTPUT_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM radio_flux", conn)
        if df.empty:
            return

        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df = df.dropna(subset=["time_tag"])
        df["time_tag"] = df["time_tag"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

        df = (
            df.sort_values("time_tag")
            .groupby("time_tag", as_index=False)
            .last()
        )

        conn.execute("DELETE FROM radio_flux")
        df.to_sql("radio_flux", conn, if_exists="append", index=False)
        conn.commit()


def _normalize_kp_dedup() -> None:
    with sqlite3.connect(OUTPUT_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM kp_index", conn)
        if df.empty:
            return

        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df = df.dropna(subset=["time_tag"])
        df["time_tag"] = df["time_tag"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")

        df = (
            df.sort_values("time_tag")
            .groupby("time_tag", as_index=False)
            .last()
        )

        conn.execute("DELETE FROM kp_index")
        df.to_sql("kp_index", conn, if_exists="append", index=False)
        conn.commit()


def _normalize_sunspot_dedup() -> None:
    with sqlite3.connect(OUTPUT_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM sunspot_numbers", conn)
        if df.empty:
            return

        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df = df.dropna(subset=["time_tag"])
        df["time_tag"] = df["time_tag"].dt.strftime("%Y-%m-%d 00:00:00+00:00")

        df = (
            df.sort_values("time_tag")
            .groupby("time_tag", as_index=False)
            .last()
        )

        conn.execute("DELETE FROM sunspot_numbers")
        df.to_sql("sunspot_numbers", conn, if_exists="append", index=False)
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
        last_dt = pd.to_datetime(row[0], utc=True)
        last_val = row[1]
        current = last_dt + timedelta(hours=1)
        while current <= now:
            conn.execute(
                "INSERT OR REPLACE INTO kp_index "
                "(time_tag, kp_index, source_type) VALUES (?, ?, ?)",
                (
                    current.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    float(last_val),
                    "forward_fill",
                ),
            )
            current += timedelta(hours=1)
        conn.commit()


def _extend_radio_flux_to_now() -> None:
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    with sqlite3.connect(OUTPUT_DB) as conn:
        row = conn.execute(
            "SELECT time_tag, observed_flux, adjusted_flux, ursi_flux "
            "FROM radio_flux ORDER BY datetime(time_tag) DESC LIMIT 1"
        ).fetchone()
        if not row:
            return
        last_dt = pd.to_datetime(row[0], utc=True)
        last_vals = row[1:]
        current = last_dt + timedelta(hours=1)
        while current <= now:
            conn.execute(
                "INSERT OR REPLACE INTO radio_flux "
                "(time_tag, observed_flux, adjusted_flux, ursi_flux, source_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    current.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                    *last_vals,
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
        last_dt = pd.to_datetime(row[0], utc=True).date()
        last_val = row[1]
        current = last_dt + timedelta(days=1)
        while current <= target_date:
            conn.execute(
                "INSERT OR REPLACE INTO sunspot_numbers "
                "(time_tag, sunspot_number, source_type) VALUES (?, ?, ?)",
                (
                    f"{current.isoformat()} 00:00:00+00:00",
                    float(last_val),
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
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True).dt.strftime(
        "%Y-%m-%d 00:00:00+00:00"
    )
    source.ingest(df, warehouse=warehouse)


def _last_table_timestamp(table: Optional[str]) -> Optional[datetime]:
    if not table:
        return None
    with sqlite3.connect(OUTPUT_DB) as conn:
        row = conn.execute(
            f"SELECT {TIME_COLUMNS[table]} FROM {table} "
            f"ORDER BY datetime({TIME_COLUMNS[table]}) DESC LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return None
        ts = pd.to_datetime(row[0], errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()


if __name__ == "__main__":
    main()
