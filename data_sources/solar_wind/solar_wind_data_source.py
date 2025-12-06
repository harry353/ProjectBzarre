from __future__ import annotations

from datetime import date, timedelta
from typing import List, Tuple

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.solar_wind.solar_wind_download_ace import download_solar_wind_ace
from data_sources.solar_wind.solar_wind_download_dscovr import download_solar_wind_dscovr
from data_sources.solar_wind.solar_wind_ingest_ace import ingest_solar_wind_ace
from data_sources.solar_wind.solar_wind_ingest_dscovr import ingest_solar_wind_dscovr
from data_sources.solar_wind.solar_wind_plot_ace import plot_solar_wind_ace
from data_sources.solar_wind.solar_wind_plot_dscovr import plot_solar_wind_dscovr

DATASET_ACE = "ace"
DATASET_DSCOVR = "dscovr"
DSCOVR_START = date(2015, 11, 1)


class SolarWindDataSource(SpaceWeatherAPI):
    """
    Downloader and parser for ACE (historic) and DSCOVR (modern) plasma data.
    """

    def _download_impl(self):
        frames: List[pd.DataFrame] = []
        for dataset, start, end in self._dataset_segments():
            if dataset == DATASET_ACE:
                frame = download_solar_wind_ace(start, end)
            else:
                frame = download_solar_wind_dscovr(start, end)

            if frame.empty:
                continue

            chunk = frame.copy()
            chunk["dataset"] = dataset
            frames.append(chunk)

        if not frames:
            return pd.DataFrame(columns=["time_tag", "density", "speed", "temperature"])

        combined = pd.concat(frames, ignore_index=True)
        combined["time_tag"] = pd.to_datetime(combined["time_tag"], errors="coerce", utc=True)
        combined = combined.dropna(subset=["time_tag"])
        return combined.sort_values("time_tag").reset_index(drop=True)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Insert solar wind rows into SQLite.
        """
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
        """
        Plot hourly averages for plasma density, speed, and temperature.
        """
        if df.empty:
            print("No solar wind data available to plot.")
            return

        if "dataset" not in df.columns:
            self._plot_single(df)
            return

        for dataset in df["dataset"].dropna().unique():
            subset = df[df["dataset"] == dataset]
            self._plot_single(subset)

    def _dataset_segments(self) -> List[Tuple[str, date, date]]:
        segments: List[Tuple[str, date, date]] = []

        if self.start_date < DSCOVR_START:
            ace_end = min(self.end_date, DSCOVR_START - timedelta(days=1))
            if self.start_date <= ace_end:
                segments.append((DATASET_ACE, self.start_date, ace_end))

        if self.end_date >= DSCOVR_START:
            dscovr_start = max(self.start_date, DSCOVR_START)
            segments.append((DATASET_DSCOVR, dscovr_start, self.end_date))

        if not segments:
            segments.append((DATASET_DSCOVR, self.start_date, self.end_date))

        return segments

    def _ingest_dataset(self, dataset: str, df: pd.DataFrame, warehouse):
        if dataset == DATASET_ACE:
            return ingest_solar_wind_ace(df, warehouse)
        return ingest_solar_wind_dscovr(df, warehouse)

    def _ingest_single(self, df: pd.DataFrame, warehouse):
        min_time = pd.to_datetime(df["time_tag"], errors="coerce").min()
        dataset = DATASET_ACE if min_time and min_time.date() < DSCOVR_START else DATASET_DSCOVR
        return self._ingest_dataset(dataset, df, warehouse)

    def _plot_single(self, df: pd.DataFrame):
        payload = df.drop(columns=["dataset"], errors="ignore")
        min_time = pd.to_datetime(payload["time_tag"], errors="coerce").min()
        if min_time and min_time.date() < DSCOVR_START:
            plot_solar_wind_ace(payload)
        else:
            plot_solar_wind_dscovr(payload)
