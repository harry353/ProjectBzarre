from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.ae.ae_download import download_ae_indices
from data_sources.ae.ae_ingest import ingest_ae
from data_sources.ae.ae_plot import plot_ae


class AEDataSource(SpaceWeatherAPI):
    """
    Access Kyoto's real-time AE/AL/AU indices via the shared API interface.
    """

    def _download_impl(self):
        return download_ae_indices(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_ae(df, warehouse)

    def plot(self, df):
        if df.empty:
            raise ValueError("Cannot plot empty AE data.")
        plot_ae(df)
