# Classification Pipeline

This directory contains the machine learning pipeline for predicting space weather storm onset probabilities. The pipeline trains models for horizons h1 through h8 (meaning 1, 2, 3, 4, 5, 6, 7, and 8 hours in advance of a storm onset), outputting the raw probabilistic predictions and calibrated probability matrices using Isotonic Regression.

## Pipeline Architecture

The end-to-end process is managed by `run_full_ml_pipeline.py`, which executes the following steps sequentially:

1. **`train_model.py` - Model Training:**
   - Ingests data from the unified `preprocessing_pipeline/merge_features/all_preprocessed_sources.db` and main phase labels.
   - Performs feature pruning to discard features that are not relevant to the prediction task.
   - Tunes hyperparameters for an `XGBClassifier` utilizing Optuna.
   - Trains and evaluates binary log-loss models for each specific predictive horizon.

2. **`export_raw_probabilities.py` - Probability Export:**
   - Uses the previously trained XGBoost models to generate a full probabilistic evaluation over all data splits (train, validation, and test).
   - Generates the `raw_probabilities.db` files inside the respective model directories.

3. **`probability_calibration.py` - Isotonic Calibration:**
   - Fits an `IsotonicRegression` calibrator on the validation split's raw probabilities.
   - Applies the learned calibration mapping jointly across all splits and horizons.
   - Generates `calibrated_probabilities.db` and the unified calibration metadata.

## Diagnostic Plotting Tools

- **`plot_probabilities.py`**: Visualizes the Dst index against cumulative storm probability over a selected time domain, demonstrating chronological hazard.
- **`plot_probability_tiles.py`**: Generates a horizonal heat-map of calibrated storm-onset interval probabilities for a specific timestamp cross-section.

## Artifacts & Model Outputs

All generated models, feature selections, and probability databases are saved outside this directory to avoid cluttering the repository. They are written to:

```text
../classification_pipeline_/horizon_models/h{X}/
├── model.json                      # Native XGBoost booster
├── selected_features.json          # Post-pruning feature list
├── summary.json                    # Optuna best-params & log-loss evaluation
├── raw_probabilities.db            # Raw outputs from export_raw_probabilities.py
└── calibrated_probabilities.db     # Final outputs post probability_calibration.py
```
