from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_DB = PROJECT_ROOT / "inference" / "regression" / "pca_regression_vector_1m.db"
IN_TABLE = "pca_inference_vector"
OUTPUT_DB = PROJECT_ROOT / "inference" / "regression" / "regression_predictions.db"

MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "rf_regime_aware_model"

HORIZONS = range(1, 2)  # predict 1..6 hours ahead
TIMESTAMP_CANDIDATES = ["timestamp", "time_tag", "date"]


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def main() -> None:
    if not IN_DB.exists():
        raise FileNotFoundError(f"Input DB not found: {IN_DB}")

    df = _load_table(IN_DB, IN_TABLE)
    if df.empty:
        raise RuntimeError(f"Input table '{IN_TABLE}' is empty.")

    # Build feature vector from numeric columns (excluding timestamp-like columns)
    time_cols = [c for c in TIMESTAMP_CANDIDATES if c in df.columns]
    feature_cols = [
        c for c in df.columns
        if c not in time_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in PCA inference table.")

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if not ts_col:
        raise RuntimeError(f"No timestamp column found in {IN_TABLE}.")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    # Keep rows with complete feature set
    df = df.dropna(subset=feature_cols)
    if df.empty:
        raise RuntimeError("No valid rows after filtering for timestamp and features.")

    X = df[feature_cols].to_numpy(dtype=float)
    ts_series = df[ts_col]

    records = []
    for h in HORIZONS:
        mu_path = MODEL_BASE / f"h{h}_calm" / "mu_model.joblib"
        sigma_path = MODEL_BASE / f"h{h}_calm" / "sigma_model.joblib"
        if not mu_path.exists() or not sigma_path.exists():
            raise FileNotFoundError(f"Missing model files for h{h}: {mu_path}, {sigma_path}")
        mu_model = joblib.load(mu_path)
        sigma_model = joblib.load(sigma_path)
        mu_pred = mu_model.predict(X).astype(float)
        sigma_pred = np.sqrt(np.exp(sigma_model.predict(X))).astype(float)

        forecast_time = ts_series + pd.Timedelta(hours=h)
        for ts_val, ft_val, mu_val, sigma_val in zip(ts_series, forecast_time, mu_pred, sigma_pred):
            records.append(
                {
                    "horizon_hours": h,
                    "timestamp": ts_val,
                    "forecast_time": ft_val,
                    "dst_mu": float(mu_val),
                    "dst_sigma": float(sigma_val),
                }
            )

    out_df = pd.DataFrame.from_records(records)
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        out_df.to_sql("predictions", conn, if_exists="replace", index=False)
    print(out_df.head())
    print(f"[OK] Wrote {len(out_df)} regression forecasts to {OUTPUT_DB} (table 'predictions').")


if __name__ == "__main__":
    main()
