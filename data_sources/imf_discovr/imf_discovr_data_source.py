from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.imf_discovr.imf_discovr_download import download_imf_discovr
from data_sources.imf_discovr.imf_discovr_ingest import ingest_imf_discovr
from data_sources.imf_discovr.imf_discovr_plot import plot_imf_discovr


class IMFDiscovrDataSource(SpaceWeatherAPI):
    """
    Retrieve DSCOVR M1M magnetic field data.
    """

    def _download_impl(self):
        """
        Download IMF data for the configured date range.
        """
        return download_imf_discovr(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Insert IMF rows into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_imf_discovr(df, warehouse)

    def plot(self, df):
        """
        Visualize IMF components.
        """
        plot_imf_discovr(df)
