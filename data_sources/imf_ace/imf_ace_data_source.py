from __future__ import annotations

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from .imf_download_ace import download_imf_ace
from .imf_ingest_ace import ingest_imf_ace
from .imf_plot_ace import plot_imf_ace


class IMFACEDataSource(SpaceWeatherAPI):
    """
    Access ACE IMF vectors for the requested date range.
    """

    def _download_impl(self):
        df = download_imf_ace(self.start_date, self.end_date)
        if df.empty:
            return df
        payload = df.copy()
        payload["time_tag"] = pd.to_datetime(payload["time_tag"], errors="coerce", utc=True)
        payload = payload.dropna(subset=["time_tag"])
        return payload.sort_values("time_tag").reset_index(drop=True)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_imf_ace(df, warehouse)

    def plot(self, df):
        if df.empty:
            raise ValueError("No ACE IMF data available to plot.")
        plot_imf_ace(df)
