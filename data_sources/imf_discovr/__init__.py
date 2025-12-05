"""
DSCOVR IMF data source package.

This package mirrors the structure used by other data sources in the
project (e.g., GOES X-ray flux) by splitting download, ingest, and plot
helpers into dedicated modules that are orchestrated by a SpaceWeatherAPI
subclass.
"""

from .imf_discovr_data_source import IMFDiscovrDataSource

__all__ = ["IMFDiscovrDataSource"]
