from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from data_sources.supermag.supermag_download import download_supermag_indices
from data_sources.supermag.supermag_ingest import ingest_supermag
from data_sources.supermag.supermag_plot import plot_supermag


class SuperMAGDataSource(SpaceWeatherAPI):
    """
    Interface to the SuperMAG SME/SMU/SML indices via the shared API base.
    """

    def __init__(self, days, logon: str, chunk_hours: int = 24):
        super().__init__(days=days)
        if not logon:
            raise ValueError("A SuperMAG logon string is required.")
        if chunk_hours <= 0:
            raise ValueError("chunk_hours must be positive.")
        self.logon = logon
        self.chunk_hours = chunk_hours

    def _download_impl(self):
        """
        Download SuperMAG indices for the configured date range.
        """
        max_days = (self.end_date - self.start_date).days + 1
        if max_days > 30:
            raise ValueError(
                "SuperMAG downloads are limited to 30 days. "
                "For larger windows please use https://supermag.jhuapl.edu/mag."
            )
        return download_supermag_indices(
            self.start_date,
            self.end_date,
            logon=self.logon,
            chunk_hours=self.chunk_hours,
        )

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Persist SuperMAG indices into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_supermag(df, warehouse)

    def plot(self, df):
        """
        Plot SuperMAG SME/SMU/SML envelopes.
        """
        if df.empty:
            raise ValueError("Cannot plot empty SuperMAG data.")
        plot_supermag(df)
