"""SpaceWeatherAPI wrapper for GOES XRS flux downloads (realtime + archive)."""

from datetime import date, datetime, timedelta, timezone
from typing import List, Tuple

import pandas as pd
import numpy as np
import requests

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from database_builder.constants import BUILD_FROM_REALTIME, REALTIME_BACKFILL_DAYS

from data_sources.xray_flux.xray_flux_goes_download import (
    GOES_OPERATIONAL_WINDOWS,
    download_xrs_goes_parallel,
)
from data_sources.xray_flux.xray_flux_goes_archive_download import (
    ARCHIVE_WINDOWS,
    download_xrs_goes_archive_parallel,
)
from data_sources.xray_flux.xray_flux_goes_ingest import ingest_xrs_goes
from data_sources.xray_flux.xray_flux_goes_plot import plot_xrs_goes

DATASET_REALTIME = "realtime"
DATASET_ARCHIVE = "archive"
SWPC_PRIMARY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
SWPC_SECONDARY_URL = "https://services.swpc.noaa.gov/json/goes/secondary/xrays-7-day.json"


def _parse_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


REALTIME_MIN_DATE = min(_parse_date(start) for start, _ in GOES_OPERATIONAL_WINDOWS.values())
ARCHIVE_MIN_DATE = min(start for _, start, _ in ARCHIVE_WINDOWS)
ARCHIVE_MAX_DATE = max(end for _, _, end in ARCHIVE_WINDOWS)


def _iter_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


class XRayFluxGOESDataSource(SpaceWeatherAPI):
    """Expose GOES XRS downloads via the common API surface."""

    def __init__(self, days, product: str = "sci"):
        super().__init__(days=days)
        product = (product or "sci").lower()
        if product not in {"sci", "ops"}:
            raise ValueError("product must be 'sci' or 'ops'")
        self.product = product

    def _download_impl(self):
        frames: List[pd.DataFrame] = []
        for dataset, start, end in self._dataset_segments():
            days = list(_iter_range(start, end))
            if not days:
                continue
            if dataset == DATASET_ARCHIVE:
                results = download_xrs_goes_archive_parallel(days)
                dataset_label = DATASET_ARCHIVE
            else:
                results = download_xrs_goes_parallel(
                    days,
                    product=self.product,
                    emit_missing=not BUILD_FROM_REALTIME,
                )
                dataset_label = DATASET_ARCHIVE

            for _, df in results:
                if df is None or df.empty:
                    continue
                chunk = df.copy()
                chunk["dataset"] = dataset_label
                chunk["source_type"] = DATASET_ARCHIVE
                frames.append(chunk)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()
        if combined.index.tz is None:
            combined.index = combined.index.tz_localize("UTC")
        else:
            combined.index = combined.index.tz_convert("UTC")
        if BUILD_FROM_REALTIME and REALTIME_BACKFILL_DAYS > 0:
            realtime_start = self.end_date - timedelta(days=REALTIME_BACKFILL_DAYS - 1)
            start_dt = datetime.combine(realtime_start, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = datetime.combine(self.end_date, datetime.max.time(), tzinfo=timezone.utc)
            swpc = _download_swpc_xrs(start_dt, end_dt)
            if not swpc.empty:
                if "time_tag" in swpc.columns:
                    swpc = swpc.set_index("time_tag")
                elif swpc.index.name != "time_tag":
                    print("[WARN] SWPC XRS payload missing time_tag; skipping realtime update.")
                    return combined
                if swpc.index.tz is None:
                    swpc.index = swpc.index.tz_localize("UTC")
                else:
                    swpc.index = swpc.index.tz_convert("UTC")
                swpc = swpc.copy()
                swpc["dataset"] = DATASET_REALTIME
                swpc["source_type"] = DATASET_REALTIME
                combined.update(swpc)
                overlap = swpc.index.intersection(combined.index)
                if not overlap.empty:
                    combined.loc[overlap, "source_type"] = "realtime"
                    combined.loc[overlap, "dataset"] = DATASET_REALTIME
                new_rows = swpc.loc[~swpc.index.isin(combined.index)]
                if not new_rows.empty:
                    combined = pd.concat([combined, new_rows], axis=0)
                    combined = combined.sort_index()
                print(f"[INFO] XRay Flux GOES realtime rows: {len(swpc)}")
        return combined

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        payload = df.drop(columns=["dataset"], errors="ignore").copy()
        return ingest_xrs_goes(payload, warehouse)

    def plot(self, df):
        if df.empty:
            print("[WARN] XRS dataframe empty. Cannot plot.")
            return

        payload = df.drop(columns=["dataset"], errors="ignore").copy()
        plot_xrs_goes(payload)

    def _dataset_segments(self) -> List[Tuple[str, date, date]]:
        segments: List[Tuple[str, date, date]] = []

        realtime_start = REALTIME_MIN_DATE
        archive_ceiling = realtime_start - timedelta(days=1)

        archive_start = max(self.start_date, ARCHIVE_MIN_DATE)
        archive_end = min(self.end_date, ARCHIVE_MAX_DATE, archive_ceiling)
        if archive_start <= archive_end:
            segments.append((DATASET_ARCHIVE, archive_start, archive_end))

        if self.end_date >= realtime_start:
            segments.append(
                (DATASET_REALTIME, max(self.start_date, realtime_start), self.end_date)
            )

        if not segments:
            segments.append((DATASET_REALTIME, self.start_date, self.end_date))

        return segments


def _download_swpc_xrs(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    primary = _download_swpc_stream(SWPC_PRIMARY_URL, start_dt, end_dt)
    secondary = _download_swpc_stream(SWPC_SECONDARY_URL, start_dt, end_dt)
    if primary.empty and secondary.empty:
        return pd.DataFrame()

    if primary.index.name == "time_tag":
        primary = primary.reset_index()
    if secondary.index.name == "time_tag":
        secondary = secondary.reset_index()

    merged = primary.merge(
        secondary,
        on="time_tag",
        how="outer",
        suffixes=("_primary", "_secondary"),
    )
    if "time_tag" not in merged.columns:
        return pd.DataFrame()
    merged = merged.sort_values("time_tag")
    merged["irradiance_xrsa1"] = merged.get("irradiance_xrsa_primary")
    merged["irradiance_xrsb1"] = merged.get("irradiance_xrsb_primary")
    merged["irradiance_xrsa2"] = merged.get("irradiance_xrsa_secondary")
    merged["irradiance_xrsb2"] = merged.get("irradiance_xrsb_secondary")
    merged["xrs_ratio"] = merged["irradiance_xrsa1"] / merged["irradiance_xrsb1"].replace(0, np.nan)
    merged["source_type"] = "realtime"

    keep = [
        "time_tag",
        "irradiance_xrsa1",
        "irradiance_xrsa2",
        "irradiance_xrsb1",
        "irradiance_xrsb2",
        "xrs_ratio",
        "source_type",
    ]
    return merged[keep].set_index("time_tag").sort_index()


def _download_swpc_stream(url: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return pd.DataFrame()
    if not isinstance(payload, list):
        return pd.DataFrame()

    df = None
    if payload and isinstance(payload[0], list):
        header = payload[0]
        rows = payload[1:]
        if isinstance(header, list) and all(isinstance(item, str) for item in header):
            df = pd.DataFrame(rows, columns=header)
    if df is None:
        df = pd.DataFrame(payload)
    if df.empty:
        return df

    time_col = None
    for candidate in ("time_tag", "time", "timestamp"):
        if candidate in df.columns:
            time_col = candidate
            break
    if not time_col:
        return pd.DataFrame()

    value_col = "flux" if "flux" in df.columns else "irradiance" if "irradiance" in df.columns else None
    if value_col is None or "energy" not in df.columns:
        return pd.DataFrame()

    df["time_tag"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["time_tag"])
    df = df[(df["time_tag"] >= start_dt) & (df["time_tag"] <= end_dt)]
    if df.empty:
        return df

    pivot = (
        df.pivot_table(index="time_tag", columns="energy", values=value_col, aggfunc="median")
        .reset_index()
        .dropna(subset=["time_tag"])
    )
    pivot["irradiance_xrsb"] = pd.to_numeric(pivot.get("0.1-0.8nm"), errors="coerce")
    pivot["irradiance_xrsa"] = pd.to_numeric(pivot.get("0.05-0.4nm"), errors="coerce")
    keep = [
        "time_tag",
        "irradiance_xrsa",
        "irradiance_xrsb",
    ]
    return pivot[keep].set_index("time_tag").sort_index()
