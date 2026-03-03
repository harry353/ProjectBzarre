# Data Sources

This directory is responsible for fetching and warehousing the raw space weather observations and forecasts upon which the entire `ProjectBzarre` machine learning pipeline is built.

## Architecture

Each specific data source is structured as a class that inherits from the core `SpaceWeatherAPI` (located at the project root). The `SpaceWeatherAPI` handles:

- **Date Range Parsing**: Consistent translation of integers, `datetime.date` objects, and `timedelta` tuples into explicit start and end temporal boundaries.
- **Downloading**: Automatic connection retry logic via the integrated `common/http.py` requests wrapper.

Once the payload is successfully downloaded, it is inserted into local SQLite databases using the lightweight `SpaceWeatherWarehouse` helper.

## Data Modules

The pipeline currently tracks and ingests the following parameters across distinct subdirectories:

- **`cme/`**: Coronal Mass Ejection observations.
- **`dst/`**: Disturbance Storm Time Index (Kyoto WDC).
- **`imf_ace/`**: Interplanetary Magnetic Field data (ACE Satellite).
- **`imf_dscovr/`**: Interplanetary Magnetic Field data (DSCOVR Satellite).
- **`kp_index/`**: Planetary K-index.
- **`radio_flux/`**: F10.7 cm Solar Radio Flux.
- **`solar_wind_ace/`**: Solar Wind Plasma parameters (ACE Satellite).
- **`solar_wind_dscovr/`**: Solar Wind Plasma parameters (DSCOVR Satellite).
- **`sunspot_number/`**: Sunspot counts/statistics.

## Module Structure

Each data module subdirectory (e.g., `dst/`) strictly adheres to a standard, four-file architectural pattern:

1. **`*_download.py`** (e.g., `dst_download.py`): Contains the targeted HTTP or FTP request logic to pull the raw payload from the external API provider.
2. **`*_ingest.py`**: Casts the raw payload into a structured Pandas DataFrame and orchestrates the SQL `INSERT` commands for the local `.db` warehouse.
3. **`*_plot.py`**: Basic helper scripts containing Matplotlib functions to quickly visualize the downloaded historical trends.
4. **`*_data_source.py`** (e.g., `dst_data_source.py`): The unifying wrapper class. Inherits from `SpaceWeatherAPI` and glues together the isolated `download`, `ingest`, and `plot` functions into a single programmatic interface.
