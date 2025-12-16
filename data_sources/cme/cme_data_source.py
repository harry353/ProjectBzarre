from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.cme.cme_download import download_cme_catalog
from data_sources.cme.cme_ingest import ingest_cme_catalog
from data_sources.cme.cme_plot import plot_cme_velocity


class CMEDataSource(SpaceWeatherAPI):
    """
    Access the CACTUS CME catalogue via the unified data source interface.
    """

    def _download_impl(self):
        return download_cme_catalog(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_cme_catalog(df, warehouse)

    def plot(self, df):
        if df.empty:
            raise ValueError("Cannot plot empty CME data.")
        plot_cme_velocity(df)
