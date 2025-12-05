from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.kp_index.kp_index_download import download_kp_index
from data_sources.kp_index.kp_index_ingest import ingest_kp_index
from data_sources.kp_index.kp_index_plot import plot_kp_index


class KpIndexDataSource(SpaceWeatherAPI):
    """
    Access GFZ's planetary Kp index through the SpaceWeatherAPI facade.
    """

    def _download_impl(self):
        """
        Download Kp index readings for the requested date range.
        """
        return download_kp_index(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Persist Kp rows into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_kp_index(df, warehouse)

    def plot(self, df):
        """
        Visualize the Kp index time series.
        """
        if df.empty:
            raise ValueError("Cannot plot empty Kp data.")
        plot_kp_index(df)
