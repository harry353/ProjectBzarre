from datetime import timedelta

import pandas as pd

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.xray_flux.xray_flux_goes_download import download_xrs_goes_parallel
from data_sources.xray_flux.xray_flux_goes_ingest import ingest_xrs_goes
from data_sources.xray_flux.xray_flux_goes_plot import plot_xrs_goes


class XRayFluxGOESDataSource(SpaceWeatherAPI):
    """
    Data source handler for GOES XRS L1b solar X-ray flux,
    resampled to 1 minute cadence.

    This class exposes:
        download()    -> returns combined dataframe for the date range
        ingest(df)    -> stores into SQLite
        plot(df)      -> quick diagnostic plot of irradiances
    """

    def __init__(self, days, product="sci"):
        """
        Initialize XRS flux source.

        Parameters
        ----------
        days : int | date | (date,date) | (date,timedelta)
            See SpaceWeatherAPI for details.
        """
        super().__init__(days=days)
        product = (product or "sci").lower()
        if product not in {"sci", "ops"}:
            raise ValueError("product must be 'sci' or 'ops'")
        self.product = product

    # ------------------------------------------------------------
    # Download
    # ------------------------------------------------------------
    def _download_impl(self):
        """
        Download and resample GOES XRS flux data for all days in the
        configured date range.

        Returns
        -------
        pandas.DataFrame
            Combined dataframe indexed by minute timestamps.
        """
        frames = []
        days = list(self.iter_days())

        for day, df in download_xrs_goes_parallel(days, product=self.product):
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")]
        out = out.sort_index()
        return out

    # ------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------
    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Insert XRS 1-minute data into SQLite.
        """

        if df.empty:
            return 0

        # FIX: ensure warehouse is always a SpaceWeatherWarehouse instance
        if warehouse is None:
            warehouse = SpaceWeatherWarehouse(db_path)

        return ingest_xrs_goes(df, warehouse)


    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    def plot(self, df):
        """
        Plot 1-minute averaged GOES XRS irradiances.

        Parameters
        ----------
        df : pandas.DataFrame
        """
        if df.empty:
            print("[WARN] XRS dataframe empty. Cannot plot.")
            return

        plot_xrs_goes(df)
