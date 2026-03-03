# Regression Pipeline

This directory handles the training and operational setup for predicting the continuous/scalar Dst index up to several hours in advance.

## Feature Engineering

Before model fitting, the pipeline applies several specific transformations configured within this directory:
- Logarithmic transformations on heavy-tailed feature candidates.
- Z-score normalization scaling.
- PCA (Principal Component Analysis) dimensionality reduction. The structure of this reduction is saved into `pca_model.joblib` so it can be repeatedly applied during operational inference.

## Modeling Strategy

The regression pipeline evaluates the severity of incoming geomagnetic storms using the following models:
- **XGBoost Quantile Regime-Aware Models** (`xgb_quantile_regime_aware_model/`): Trees parameterized to output probabilistic quantile regressions (e.g., q10, q50, q90) split across different behavioral regimes (calm vs. storm).

## Execution

The `run_full_regression.py` orchestrator script executes the transformation diagnostics, applies the log/PCA matrices, and invokes the model training loops across the defined prediction horizons.
