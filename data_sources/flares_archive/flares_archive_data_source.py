"""SpaceWeatherAPI adapter for GOES flare archive data."""

from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.flares_archive.flares_archive_download import download_flares_archive
from data_sources.flares_archive.flares_archive_ingest import ingest_flares_archive
from data_sources.flares_archive.flares_archive_plot import plot_flares_archive


class FlaresArchiveDataSource(SpaceWeatherAPI):
    """Expose archive flare data using the shared API surface."""

    def _download_impl(self):
        return download_flares_archive(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        if df.empty:
            return 0
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_flares_archive(df, warehouse)

    def plot(self, df):
        title_suffix = f"{self.start_date} -> {self.end_date}"
        plot_flares_archive(df, title_suffix=title_suffix)
