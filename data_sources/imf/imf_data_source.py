"""SpaceWeatherAPI wrapper for IMF data from ACE (pre-2015) and DSCOVR."""

from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.imf.imf_download_ace import download_imf_ace
from data_sources.imf.imf_ingest_ace import ingest_imf_ace
from data_sources.imf.imf_plot_ace import plot_imf_ace
from data_sources.imf.imf_download_discovr import download_imf_discovr
from data_sources.imf.imf_ingest_discovr import ingest_imf_discovr
from data_sources.imf.imf_plot_discovr import plot_imf_discovr

DATASET_ACE = "ace"
DATASET_DISCOVR = "dscovr"
DISCOVR_START = date(2015, 11, 1)


class IMFACEDataSource(SpaceWeatherAPI):
    """
    Access IMF vectors from ACE (before Nov 2015) and DSCOVR (after).
    """

    def _download_impl(self):
        frames: List[pd.DataFrame] = []
        for dataset, start, end in self._dataset_segments():
            if dataset == DATASET_ACE:
                frame = download_imf_ace(start, end)
            else:
                frame = download_imf_discovr(start, end)
            if frame.empty:
                continue
            chunk = frame.copy()
            chunk["dataset"] = dataset
            frames.append(chunk)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        if "time_tag" in combined.columns:
            combined["time_tag"] = pd.to_datetime(combined["time_tag"], errors="coerce")
            combined = combined.sort_values("time_tag").reset_index(drop=True)
        return combined

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)

        if "dataset" not in df.columns:
            return self._ingest_single(df, warehouse)

        total = 0
        for dataset in df["dataset"].dropna().unique():
            subset = df[df["dataset"] == dataset].drop(columns=["dataset"])
            total += self._ingest_dataset(dataset, subset, warehouse)
        return total

    def plot(self, df):
        if df.empty:
            print("No IMF data available to plot.")
            return

        if "dataset" not in df.columns:
            self._plot_single(df)
            return

        for dataset in df["dataset"].dropna().unique():
            subset = df[df["dataset"] == dataset]
            self._plot_single(subset)

    def _dataset_segments(self) -> List[Tuple[str, date, date]]:
        segments: List[Tuple[str, date, date]] = []

        if self.start_date < DISCOVR_START:
            ace_end = min(self.end_date, DISCOVR_START - timedelta(days=1))
            if self.start_date <= ace_end:
                segments.append((DATASET_ACE, self.start_date, ace_end))

        if self.end_date >= DISCOVR_START:
            discovr_start = max(self.start_date, DISCOVR_START)
            segments.append((DATASET_DISCOVR, discovr_start, self.end_date))

        if not segments:
            segments.append((DATASET_DISCOVR, self.start_date, self.end_date))

        return segments

    def _ingest_dataset(self, dataset: str, df: pd.DataFrame, warehouse):
        if dataset == DATASET_ACE:
            return ingest_imf_ace(df, warehouse)
        return ingest_imf_discovr(df, warehouse)

    def _ingest_single(self, df: pd.DataFrame, warehouse):
        if {"bx_gse", "by_gse", "bz_gse"}.issubset(df.columns):
            return ingest_imf_ace(df, warehouse)
        return ingest_imf_discovr(df, warehouse)

    def _plot_single(self, df: pd.DataFrame):
        payload = df.drop(columns=["dataset"], errors="ignore").copy()
        if "time_tag" in payload.columns:
            payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce")
            ts_min = payload["time_tag"].min()
            ts_max = payload["time_tag"].max()
            print(f"Plotting timestamps from {ts_min} to {ts_max}")
        if {"bx_gse", "by_gse", "bz_gse"}.issubset(payload.columns):
            plot_imf_ace(payload)
        else:
            plot_imf_discovr(payload)
