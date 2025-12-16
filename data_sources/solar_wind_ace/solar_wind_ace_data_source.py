from __future__ import annotations

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from .solar_wind_download_ace import download_solar_wind_ace
from .solar_wind_ingest_ace import ingest_solar_wind_ace
from .solar_wind_plot_ace import plot_solar_wind_ace


class SolarWindACEDataSource(SpaceWeatherAPI):
    """
    Downloader and plotter for ACE solar wind plasma data only.
    """

    def _download_impl(self):
        df = download_solar_wind_ace(self.start_date, self.end_date)
        if df.empty:
            return df
        df = df.copy()
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df = df.dropna(subset=["time_tag"])
        return df.sort_values("time_tag").reset_index(drop=True)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_solar_wind_ace(df, warehouse)

    def plot(self, df):
        if df.empty:
            raise ValueError("No ACE solar wind data available to plot.")
        plot_solar_wind_ace(df)
