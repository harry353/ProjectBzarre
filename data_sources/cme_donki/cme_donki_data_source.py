from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.cme_donki.cme_donki_download import download_cme
from data_sources.cme_donki.cme_donki_ingest import ingest_cme
from data_sources.cme_donki.cme_donki_plot import plot_cme


class CMEDataSource(SpaceWeatherAPI):
    """
    Access NASA's DONKI CMEAnalysis feed through the SpaceWeatherAPI.
    """

    def _download_impl(self):
        """
        Download CMEAnalysis entries for the configured time range.
        """
        return download_cme(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Persist CME rows into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_cme(df, warehouse)

    def plot(self, df):
        """
        Visualize CME counts per day and their speeds.
        """
        plot_cme(df)
