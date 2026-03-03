# Preprocessing Pipeline

This directory is responsible for transforming raw space weather data (from `space_weather.db`) into clean, time-aligned, and feature-engineered datasets suitable for machine learning (`all_preprocessed_sources.db`).

## Architecture

The pipeline is organized into sub-modules handling logic specific to different data sources (e.g., `imf_solar_wind/`, `dst/`, `sunspot_number/`). Each sub-module reads raw data, handles missing values, resamples it to a consistent temporal frequency, and applies target labels when necessary.

Crucially, the preprocessing pipeline guarantees no data leakage by enforcing shared chronologies. Environment variables dictate strict temporal boundaries for the Train, Validation, and Test splits.

## Execution

The main entrypoint for this directory is `run_full_preprocessing_pipeline.py`. 

This script:
1. Sets the project-wide train/validation/test date ranges via environment variables.
2. Invokes the individual `run_pipeline.py` scripts nested inside each sub-module.
3. Finalizes the process by executing `merge_features.py` (located in the `merge_features/` directory), which performs a massive outer join across all preprocessed data sources on their timestamps, yielding the final `all_preprocessed_sources.db` asset.
