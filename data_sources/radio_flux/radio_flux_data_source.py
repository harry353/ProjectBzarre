from space_weather_api import SpaceWeatherAPI
from space_weather_warehouse import SpaceWeatherWarehouse

from data_sources.radio_flux.radio_flux_download import download_radio_flux
from data_sources.radio_flux.radio_flux_ingest import ingest_radio_flux
from data_sources.radio_flux.radio_flux_plot import plot_radio_flux


class RadioFluxDataSource(SpaceWeatherAPI):
    """
    Access Penticton F10.7 radio flux readings using the shared API.
    """
    RUN_IN_THREADPOOL = False

    def _download_impl(self):
        # Delegate to the dedicated download helper, passing the date range from the base class
        return download_radio_flux(self.start_date, self.end_date)

    def ingest(self, df, warehouse=None, db_path="space_weather.db"):
        # Nothing to persist if the dataframe is empty
        if df.empty:
            return 0

        # Fall back to a default warehouse pointed at the standard database file
        warehouse = warehouse or SpaceWeatherWarehouse(db_path)
        return ingest_radio_flux(df, warehouse)

    def plot(self, df):
        # Guard against being called with no data before handing off to the plot helper
        if df.empty:
            raise ValueError("Cannot plot empty radio flux data.")
        plot_radio_flux(df)
