"""Unified CME data source that stitches LASCO and DONKI feeds."""

from __future__ import annotations

from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.cme.cme_lasco_download import download_cme_lasco
from data_sources.cme.cme_lasco_ingest import ingest_cme_lasco
from data_sources.cme.cme_lasco_plot import plot_cme_lasco
from data_sources.cme.cme_donki_download import download_cme
from data_sources.cme.cme_donki_ingest import ingest_cme
from data_sources.cme.cme_donki_plot import plot_cme

DATASET_LASCO = "lasco"
DATASET_DONKI = "donki"
DONKI_START = date(2010, 1, 1)


class CMEDataSource(SpaceWeatherAPI):
    """
    Access LASCO CME entries for historical ranges and switch to the
    NASA DONKI CMEAnalysis service for modern data.
    """

    def _download_impl(self):
        frames: List[pd.DataFrame] = []

        for dataset, start, end in self._dataset_segments():
            if dataset == DATASET_LASCO:
                frame = download_cme_lasco(start, end)
            else:
                frame = download_cme(start, end)

            if frame.empty:
                continue

            chunk = frame.copy()
            chunk["dataset"] = dataset
            frames.append(chunk)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        return self._sort_by_timestamp(combined)

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
            print("No CME data available to plot.")
            return

        if "dataset" not in df.columns:
            self._plot_single(df)
            return

        for dataset in df["dataset"].dropna().unique():
            subset = df[df["dataset"] == dataset]
            self._plot_single(subset)

    def _dataset_segments(self) -> List[Tuple[str, date, date]]:
        segments: List[Tuple[str, date, date]] = []

        if self.start_date < DONKI_START:
            lasco_end = min(self.end_date, DONKI_START - timedelta(days=1))
            if self.start_date <= lasco_end:
                segments.append((DATASET_LASCO, self.start_date, lasco_end))

        if self.end_date >= DONKI_START:
            donki_start = max(self.start_date, DONKI_START)
            segments.append((DATASET_DONKI, donki_start, self.end_date))

        if not segments:
            segments.append((DATASET_DONKI, self.start_date, self.end_date))

        return segments

    def _ingest_dataset(self, dataset: str, df: pd.DataFrame, warehouse):
        if dataset == DATASET_LASCO:
            return ingest_cme_lasco(df, warehouse)
        return ingest_cme(df, warehouse)

    def _ingest_single(self, df: pd.DataFrame, warehouse):
        if "Datetime" in df.columns:
            return ingest_cme_lasco(df, warehouse)
        return ingest_cme(df, warehouse)

    def _plot_single(self, df: pd.DataFrame):
        payload = df.drop(columns=["dataset"], errors="ignore").copy()
        if "Datetime" in payload.columns:
            payload["Datetime"] = pd.to_datetime(payload["Datetime"], errors="coerce", utc=True)
            plot_cme_lasco(payload)
        else:
            payload["time21_5"] = pd.to_datetime(payload.get("time21_5"), errors="coerce", utc=True)
            plot_cme(payload)

    def _sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        payload = df.copy()
        payload["sort_time"] = pd.NaT

        if "Datetime" in payload.columns:
            payload["Datetime"] = (
                pd.to_datetime(payload["Datetime"], errors="coerce", utc=True)
                .dt.tz_convert(None)
            )
        if "time21_5" in payload.columns:
            payload["time21_5"] = (
                pd.to_datetime(payload["time21_5"], errors="coerce", utc=True)
                .dt.tz_convert(None)
            )

        if "dataset" in payload.columns:
            mask_lasco = payload["dataset"] == DATASET_LASCO
            mask_donki = payload["dataset"] == DATASET_DONKI
            if "Datetime" in payload.columns:
                payload.loc[mask_lasco, "sort_time"] = payload.loc[mask_lasco, "Datetime"]
            if "time21_5" in payload.columns:
                payload.loc[mask_donki, "sort_time"] = payload.loc[mask_donki, "time21_5"]
        else:
            if "Datetime" in payload.columns:
                payload["sort_time"] = payload["Datetime"]
            elif "time21_5" in payload.columns:
                payload["sort_time"] = payload["time21_5"]

        payload = payload.sort_values("sort_time").drop(columns=["sort_time"])
        return payload.reset_index(drop=True)
