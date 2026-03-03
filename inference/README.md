# Inference Pipeline

This directory provides the end-to-end execution pipeline that fetches live space weather data, preprocesses it, and generates live predictions using the pre-trained models.

## Architecture

Instead of completely rebuilding the historical warehouse (`space_weather.db`) upon every run, this directory uses localized snapshots to ensure low latency. 

### Key Orchestrators
- **`run_full_inference.py`**: The master entrypoint that executes the full operational sequence: updating the live snapshot, preprocessing the feature vectors, and executing both classification and regression inference scripts chronologically.
- **`update_space_weather_last_6m.py`**: Data fetcher that queries the APIs in `data_sources/` for exactly the trailing 6 months of metrics. The data is cached locally as `space_weather_last_6m.db`.

### Fallback Processing
If the primary IMF data sources happen to be offline (can often be the case with DSCOVR), the inference pipeline attempts to salvage the run using trailing fallback data from SWPC:
- `backup_swpc_imf.py`: Pulls fallback magnetic field features.
- `insert_swpc_imf_backup.py`: Injects fallback data into the pipeline if primary fetches leave a temporal gap.

## Execution Modules

Once `space_weather_last_6m.db` is built and preprocessed, the pipeline hands off the features to the two core inferencing modules:

- **Classification (`classification/`)**: Generates calibrated probabilities regarding storm onset (Dst < -20nT) across 8 distinct horizons spanning up to 8 hours ahead. Outputs its forecasts to `classification_predictions.db`.
- **Regression (`regression/`)**: Generates regime-aware probabilistic quantile forecasts (p10, p50, p90) predicting the exact severity of the Dst index out to 6 hours ahead. Outputs its results to `regression_predictions.db`.

## Visualization Output

The `combined_plot/` sub-directory evaluates these dual `.db` outputs and generates the primary dashboard graphics used to monitor the models' live forecasting accuracy.
