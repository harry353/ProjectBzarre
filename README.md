# ProjectBzarre

End-to-end space weather ML pipeline for data ingestion, preprocessing, label generation, model training, and probability calibration.

## Repository layout
- `data_sources/`: data download and collection scripts
- `database_builder/`: raw data warehouse and table construction
- `preprocessing_pipeline/`: feature engineering, aggregation, splits, normalization, labels, and final merge
- `modeling_pipeline/`: training and evaluation scripts (multi-horizon)
- `modeling_pipeline_daily/`: legacy daily modeling utilities and plots
- `probability_calibration/`: calibration DB builder, regime-aware isotonic calibration, and plots
- `tests/`: test suite

## Typical workflow
1. Build or refresh raw data sources.
2. Run preprocessing pipelines per data source.
3. Merge final datasets into a unified SQLite DB.
4. Train models for horizons 1–8.
5. Build calibration DB and fit regime-aware calibrators.
6. Plot diagnostics as needed.

## Key artifacts
- `preprocessing_pipeline/final/all_preprocessed_sources.db`: merged feature/label dataset
- `modeling_pipeline/output_h{X}/`: per-horizon models and diagnostics
- `probability_calibration/validation_calibration.db`: calibration dataset
- `probability_calibration/calibration_h{X}/`: per-horizon isotonic calibrators + metadata

## Running
Most scripts are executable as standalone Python files. Example:
```
/bin/python3 preprocessing_pipeline/final/merge_final_datasets.py
/bin/python3 modeling_pipeline/train_model.py
/bin/python3 probability_calibration/build_calibration_db.py
/bin/python3 probability_calibration/regime_aware_calibration.py
```

## Notes
- Databases are SQLite and live under their respective pipeline directories.
- Many stages rely on environment variables for split windows and aggregation cadence.
- Horizon selection for training and calibration is handled by constants in the scripts.
