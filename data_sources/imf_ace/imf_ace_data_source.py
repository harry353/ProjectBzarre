from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.imf_ace.imf_ace_download import download_imf_ace
from data_sources.imf_ace.imf_ace_ingest import ingest_imf_ace
from data_sources.imf_ace.imf_ace_plot import plot_imf_ace


class IMFACEDataSource(SpaceWeatherAPI):
    """
    Access ACE/MAG IMF vectors using the shared data-source interface.
    """

    def _download_impl(self):
        """
        Download ACE/MAG H3 data for the configured date range.
        """
        return download_imf_ace(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Store ACE/MAG series inside SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_imf_ace(df, warehouse)

    def plot(self, df):
        """
        Plot IMF components and total field magnitude.
        """
        if df.empty:
            raise ValueError("Cannot plot empty IMF ACE data.")
        plot_imf_ace(df)
