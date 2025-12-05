from __future__ import annotations

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.cme_lasco.cme_lasco_download import download_cme_lasco
from data_sources.cme_lasco.cme_lasco_ingest import ingest_cme_lasco
from data_sources.cme_lasco.cme_lasco_plot import plot_cme_lasco


class CMELASCODataSource(SpaceWeatherAPI):
    """
    Access LASCO CME entries from the CDAW catalog.
    """

    def _download_impl(self):
        """
        Retrieve LASCO CME rows for the configured date range.
        """
        return download_cme_lasco(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Persist LASCO CME rows into SQLite storage.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_cme_lasco(df, warehouse)

    def plot(self, df):
        """
        Visualize LASCO CME linear speeds for the current range.
        """
        title_suffix = f"{self.start_date} → {self.end_date}"
        plot_cme_lasco(df, title_suffix=title_suffix)
