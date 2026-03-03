# Tests

This directory contains the automated Pytest suite for verifying the integrity and stability of the project.

## Test Categories

The checks are primarily focused on verifying the `SpaceWeatherAPI` data ingestion endpoints located in the `data_sources/` directory.

- **`test_all_sources.py`**: A general, comprehensive check ensuring that all configured providers download and parse basic timeframes correctly.
- **Unit Validations**: Specific test scripts for individual fetchers, ensuring edge cases and unique data formats are handled properly:
  - `test_cme.py`
  - `test_dst.py`
  - `test_flares.py`
  - `test_solar_wind_ace.py`, etc.

