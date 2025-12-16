from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse
from .sw_comp_download import download_sw_comp
from .sw_comp_ingest import ingest_sw_comp
from .sw_comp_plot import plot_sw_comp


class SWCompDataSource(SpaceWeatherAPI):
    """
    Access ACE/SWICS heavy ion composition ratios via the shared interface.
    """

    def _download_impl(self):
        """
        Download ACE/SWICS composition ratios for the configured date range.
        """
        return download_sw_comp(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        """
        Store ACE/SWICS composition ratios into SQLite.
        """
        if df.empty:
            return 0

        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_sw_comp(df, warehouse)

    def plot(self, df):
        """
        Plot ACE/SWICS composition ratios.
        """
        if df.empty:
            raise ValueError("Cannot plot empty SW composition data.")
        plot_sw_comp(df)
