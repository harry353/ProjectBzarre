from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from archived_sources.solar_flare_donki.solar_flare_donki_download import download_flares
from archived_sources.solar_flare_donki.solar_flare_donki_ingest import ingest_flares
from archived_sources.solar_flare_donki.solar_flare_donki_plot import plot_flares


class SolarFlareDonkiDataSource(SpaceWeatherAPI):
    """
    Access the NASA DONKI solar flare catalog.
    """

    def _download_impl(self):
        """
        Fetch flare events for the configured time range.
        """
        return download_flares(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Insert flare rows into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_flares(df, warehouse)

    def plot(self, df):
        """
        Plot flare occurrences colored by their class.
        """
        title_suffix = f"{self.start_date} → {self.end_date}"
        plot_flares(df, title_suffix=title_suffix)
