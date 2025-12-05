from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.solar_wind.solar_wind_download import download_solar_wind
from data_sources.solar_wind.solar_wind_ingest import ingest_solar_wind
from data_sources.solar_wind.solar_wind_plot import plot_solar_wind


class SolarWindDataSource(SpaceWeatherAPI):
    """
    Downloader and parser for DSCOVR F1M plasma.
    """

    def _download_impl(self):
        """
        Download DSCOVR F1M plasma NetCDF files in parallel.
        """
        return download_solar_wind(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Insert DSCOVR F1M rows into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_solar_wind(df, warehouse)

    def plot(self, df):
        """
        Plot hourly averages for plasma density, speed, and temperature.
        """
        plot_solar_wind(df)
