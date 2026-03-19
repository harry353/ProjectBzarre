from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.dst.dst_download import download_dst
from data_sources.dst.dst_ingest import ingest_dst
from data_sources.dst.dst_plot import plot_dst


class DstDataSource(SpaceWeatherAPI):
    """
    Access the Kyoto WDC Dst index using the shared SpaceWeatherAPI interface.
    """

    def _download_impl(self):
        # Delegate to the module-level download function using the inherited date range.
        return download_dst(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        # Skip ingestion silently when the downloaded DataFrame is empty.
        if df.empty:
            return 0

        # Create a default warehouse if the caller did not supply one.
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_dst(df, warehouse)

    def plot(self, df):
        if df.empty:
            raise ValueError("Cannot plot empty Dst data.")
        plot_dst(df)
