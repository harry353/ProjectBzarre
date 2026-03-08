from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IN_DB = PROJECT_ROOT / "inference" / "regression" / "pca_regression_vector_1m.db"
IN_TABLE = "pca_inference_vector"
DST_DB = PROJECT_ROOT / "inference" / "preprocessed_vector_1m.db"
DST_TABLE = "inference_vector"
OUTPUT_DB = PROJECT_ROOT / "inference" / "regression" / "regression_predictions.db"

MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"

HORIZONS = range(1, 7)  # predict 1..6 hours ahead
QUANTILES = (0.1, 0.5, 0.9)
DST_THRESHOLD = -20.0
TIMESTAMP_CANDIDATES = ["timestamp", "time_tag", "date"]


# Helper to load a SQLite database table into a pandas DataFrame
def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


# Search for valid timestamp column names in the provided DataFrame
def _detect_ts(df: pd.DataFrame) -> str | None:
    for c in TIMESTAMP_CANDIDATES:
        if c in df.columns:
            return c
    return None


# Search for Dst-related column names or fallback to the first numeric column
def _detect_dst(df: pd.DataFrame) -> str | None:
    for c in ("h1", "dst", "Dst", "dst_value", "dst_dst"):
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


# Load XGBoost quantile models and generate predictions for a given regime and horizon
def _predict_set(model_cache: dict, reg: str, h: int, x: np.ndarray) -> dict[float, float] | None:
    preds = {}
    for q in QUANTILES:
        key = (reg, h, q)
        # Use lazy loading to keep the memory footprint low
        if key not in model_cache:
            model_path = MODEL_BASE / f"h{h}_{reg}" / f"q{q}" / "model.joblib"
            if not model_path.exists():
                return None
            model_cache[key] = joblib.load(model_path)
        model = model_cache[key]
        preds[q] = float(model.predict(x)[0])
    
    # Enforce monotonic ordering (q0.1 <= q0.5 <= q0.9)
    # This corrects crossing quantiles which can occur in independent quantile regression
    ordered = {}
    last = -np.inf
    for q in sorted(preds):
        val = preds[q]
        if val < last:
            val = last
        ordered[q] = val
        last = val
    return ordered


# Main orchestration flow for generating regression-based Dst forecasts
def main() -> None:
    # 1. Verification of upstream PCA and preprocessing artifacts
    if not IN_DB.exists():
        raise FileNotFoundError(f"Input DB not found: {IN_DB}")
    if not DST_DB.exists():
        raise FileNotFoundError(f"DST DB not found: {DST_DB}")

    # 2. Loading features and target targets
    feat_df = _load_table(IN_DB, IN_TABLE)
    if feat_df.empty:
        raise RuntimeError(f"Input table '{IN_TABLE}' is empty.")

    dst_df = _load_table(DST_DB, DST_TABLE)
    ts_col_feat = _detect_ts(feat_df)
    ts_col_dst = _detect_ts(dst_df)
    dst_col = _detect_dst(dst_df)
    
    if not ts_col_feat or not ts_col_dst or not dst_col:
        raise RuntimeError("Could not detect timestamp/DST columns for regime detection.")

    # 3. Time-alignment and merging
    # Clean and align timestamps between PCA features and raw Dst for regime detection
    feat_df[ts_col_feat] = pd.to_datetime(feat_df[ts_col_feat], utc=True, errors="coerce").dt.tz_localize(None)
    dst_df[ts_col_dst] = pd.to_datetime(dst_df[ts_col_dst], utc=True, errors="coerce").dt.tz_localize(None)

    dst_df = dst_df[[ts_col_dst, dst_col]].rename(columns={ts_col_dst: ts_col_feat, dst_col: "dst_dst"})
    merged = (
        feat_df
        .merge(dst_df, on=ts_col_feat, how="left")
    )
    merged = merged.dropna(subset=[ts_col_feat, "dst_dst"])
    
    # Identify numeric features for the models, ensuring 'dst_dst' is included as a lagged feature
    feature_cols = [
        c for c in merged.columns
        if c != ts_col_feat and pd.api.types.is_numeric_dtype(merged[c])
    ]
    merged = merged.dropna(subset=feature_cols)
    if merged.empty:
        raise RuntimeError("No valid rows after merging features with dst_dst.")

    ts_series = merged[ts_col_feat]
    X = merged[feature_cols].to_numpy(dtype=np.float32)

    # 4. Sequential Prediction Loop
    model_cache: dict[tuple[str, int, float], object] = {}
    records = []
    for row_idx, (ts_val, dst_val) in enumerate(zip(ts_series, merged["dst_dst"])):
        # Pivot models based on the current geomagnetic regime (threshold-based)
        regime = "storm" if dst_val <= DST_THRESHOLD else "calm"
        x_row = X[row_idx].reshape(1, -1)
        
        # Iteratively predict all horizons (e.g., 1h, 2h, ... up to 6h ahead)
        for h in HORIZONS:
            preds = _predict_set(model_cache, regime, h, x_row)
            if preds is None:
                continue
            forecast_time = ts_val + pd.Timedelta(hours=h)
            records.append(
                {
                    "horizon_hours": h,
                    "timestamp": ts_val,
                    "forecast_time": forecast_time,
                    "regime": regime,
                    "q10": preds[0.1],
                    "q50": preds[0.5],
                    "q90": preds[0.9],
                }
            )

    # 5. Result Persistence
    out_df = pd.DataFrame.from_records(records)
    if out_df.empty:
        raise RuntimeError("No regression forecasts generated.")

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        out_df.to_sql("predictions", conn, if_exists="replace", index=False)
    
    print(out_df.head())
    print(f"[OK] Wrote {len(out_df)} regression forecasts to {OUTPUT_DB} (table 'predictions').")


if __name__ == "__main__":
    main()
