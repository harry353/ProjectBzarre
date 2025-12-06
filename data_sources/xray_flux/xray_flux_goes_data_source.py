"""SpaceWeatherAPI wrapper for GOES XRS flux downloads (realtime + archive)."""

from datetime import date, datetime, timedelta
from typing import List, Tuple

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

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
            else:
                results = download_xrs_goes_parallel(days, product=self.product)

            for _, df in results:
                if df is None or df.empty:
                    continue
                chunk = df.copy()
                chunk["dataset"] = dataset
                frames.append(chunk)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="first")]
        return combined.sort_index()

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
