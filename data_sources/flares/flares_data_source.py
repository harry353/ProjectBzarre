"""SpaceWeatherAPI wrapper for GOES flare downloads (realtime + archive)."""

from datetime import date, datetime, timedelta
from typing import List, Tuple

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.flares.flares_download import (
    DATASET_ARCHIVE,
    DATASET_REALTIME,
    GOES_ARCHIVE_RANGES,
    GOES_OPERATIONAL_WINDOWS,
    download_flares,
)
from data_sources.flares.flares_ingest import ingest_flares
from data_sources.flares.flares_plot import plot_flares


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


REALTIME_MIN_DATE = min(start for start, _ in GOES_OPERATIONAL_WINDOWS.values())
ARCHIVE_MIN_DATE = min(_parse_date(start) for start, _ in GOES_ARCHIVE_RANGES.values())
ARCHIVE_MAX_DATE = max(_parse_date(end) for _, end in GOES_ARCHIVE_RANGES.values())


class FlaresDataSource(SpaceWeatherAPI):
    """Expose GOES flare downloads via the common API surface."""

    def _download_impl(self):
        frames: List[pd.DataFrame] = []
        for dataset, start, end in self._dataset_segments():
            frame = download_flares(start, end, dataset=dataset)
            if frame.empty:
                continue
            chunk = frame.copy()
            chunk["dataset"] = dataset
            frames.append(chunk)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        if "dataset" not in df.columns:
            return ingest_flares(df, warehouse, DATASET_REALTIME)

        total = 0
        for dataset in df["dataset"].dropna().unique():
            subset = df[df["dataset"] == dataset].drop(columns=["dataset"])
            total += ingest_flares(subset, warehouse, dataset)
        return total

    def plot(self, df):
        if df.empty:
            print("No GOES flare data available to plot.")
            return

        title_suffix = f"{self.start_date} -> {self.end_date}"
        if "dataset" not in df.columns:
            plot_flares(df, dataset=DATASET_REALTIME, title_suffix=title_suffix)
            return

        for dataset in df["dataset"].dropna().unique():
            subset = df[df["dataset"] == dataset]
            plot_flares(subset, dataset=dataset, title_suffix=title_suffix)

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
