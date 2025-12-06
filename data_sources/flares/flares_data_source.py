"""SpaceWeatherAPI wrapper for GOES flare summary downloads."""

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.flares.flares_download import download_flares
from data_sources.flares.flares_ingest import ingest_flares
from data_sources.flares.flares_plot import plot_flares


class FlaresDataSource(SpaceWeatherAPI):
    """Expose GOES flare summary downloads via the common API surface."""

    def _download_impl(self):
        return download_flares(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_flares(df, warehouse)

    def plot(self, df):
        title_suffix = f"{self.start_date} -> {self.end_date}"
        plot_flares(df, title_suffix=title_suffix)
