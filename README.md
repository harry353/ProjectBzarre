# ProjectBzarre

End-to-end space weather ML pipeline for data ingestion, preprocessing, label generation, model training (regression and classification), and inference.

## Project Architecture

The project is structured into distinct pipeline stages:

1. **`data_sources/`**: Scripts to download and collect raw space weather data from various sources (ACE, DSCOVR, SWPC, etc.).
2. **`database_builder/`**: SQLite utilities to handle raw data warehousing and table construction.
3. **`preprocessing_pipeline/`**: Feature engineering, aggregation, splits, normalization, label target generation, and merging into unified datasets.
4. **`regression_pipeline/` & `classification_pipeline/`**: Training, evaluation, and modeling scripts for multi-horizon forecasting and probability calculation.
5. **`inference/`**: End-to-end inference execution, taking live/recent data, formatting it, and running the pre-trained models.
6. **`common/`**: Shared utilities (e.g., logging, HTTP requests).
7. **`tests/`**: Project tests.

## Setup & Environment

The project requires several scientific and machine learning libraries (e.g., NumPy, Pandas, Astropy, Sunpy, Scikit-Learn, XGBoost). 
You can set up the environment using Conda with the provided `environment.yml` or pip with `requirements.txt`.

Example using Conda:
```bash
conda env create -f environment.yml
conda activate projectbzarre
```

## Running the Pipelines

Most major stages of the pipeline have top-level runner scripts that execute the components in the correct order. Example entry points:

- **Preprocessing**: `python3 preprocessing_pipeline/run_full_preprocessing_pipeline.py`
- **Regression Modeling**: `python3 regression_pipeline/run_full_regression.py`
- **Inference**: `python3 inference/run_full_inference.py`

## Notes
- Databases are primarily Local SQLite (`.db`) files and reside under their respective pipeline directories.
- Many stages rely on environment variables for configuration, such as defining split windows and aggregation cadences (e.g., `PREPROC_SPLIT_TRAIN_START`, `PREPROC_AGG_FREQ`).
- Horizon selection for training is generally handled by constants within the specific modeling scripts.
