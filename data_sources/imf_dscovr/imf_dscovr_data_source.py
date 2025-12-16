from __future__ import annotations

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from .imf_download_discovr import download_imf_discovr
from .imf_ingest_discovr import ingest_imf_discovr
from .imf_plot_discovr import plot_imf_discovr


class IMFDSCOVRDataSource(SpaceWeatherAPI):
    """
    Access DSCOVR IMF vectors for the requested date range.
    """

    def _download_impl(self):
        df = download_imf_discovr(self.start_date, self.end_date)
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
        return ingest_imf_discovr(df, warehouse)

    def plot(self, df):
        if df.empty:
            raise ValueError("No DSCOVR IMF data available to plot.")
        plot_imf_discovr(df)
