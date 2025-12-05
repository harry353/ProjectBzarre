import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.xray_flux.xray_flux_goes_archive_download import (
    download_xrs_goes_archive_parallel,
)
from data_sources.xray_flux.xray_flux_goes_ingest import ingest_xrs_goes
from data_sources.xray_flux.xray_flux_goes_plot import plot_xrs_goes


class XRayFluxGOESArchiveDataSource(SpaceWeatherAPI):
    """Access historical GOES XRS 1-minute archive products."""

    def _download_impl(self):
        frames = []
        for day, df in download_xrs_goes_archive_parallel(self.iter_days()):
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")]
        return out.sort_index()

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_xrs_goes(df, warehouse)

    def plot(self, df):
        if df.empty:
            print("[WARN] XRS archive dataframe empty. Cannot plot.")
            return

        plot_xrs_goes(df)
