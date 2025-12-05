from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.sunspot_number.sunspot_number_download import (
    download_sunspot_numbers,
)
from data_sources.sunspot_number.sunspot_number_ingest import (
    ingest_sunspot_numbers,
)
from data_sources.sunspot_number.sunspot_number_plot import plot_sunspot_numbers


class SunspotNumberDataSource(SpaceWeatherAPI):
    """
    Access GFZ's daily sunspot number index through the SpaceWeatherAPI facade.
    """

    def _download_impl(self):
        """
        Download sunspot number rows for the requested time range.
        """
        return download_sunspot_numbers(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Persist sunspot numbers into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_sunspot_numbers(df, warehouse)

    def plot(self, df):
        """
        Visualize the sunspot number time series.
        """
        if df.empty:
            raise ValueError("Cannot plot empty sunspot number data.")
        plot_sunspot_numbers(df)
