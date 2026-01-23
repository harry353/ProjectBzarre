from __future__ import annotations

from pathlib import Path

import pandas as pd
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DST_DB = PROJECT_ROOT / "inference" / "preprocessed_vector_1m.db"
DST_TABLE = "inference_vector"
FEATURE_DB = PROJECT_ROOT / "inference" / "regression" / "pca_regression_vector_1m.db"
FEATURE_TABLE = "pca_inference_vector"
MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"

PROB_DB = PROJECT_ROOT / "inference" / "classification" / "classification_predictions.db"
PROB_TABLE = "predictions"
PROB_TS_COL = "timestamp"

STORM_COLOR = "#ff7f7f"
CALM_COLOR = "#1f77b4"

HISTORY_HOURS = 24 * 14
FUTURE_HOURS = 6
INSET_PAD_HOURS_BEFORE = 3
INSET_PAD_HOURS_AFTER = 1
QUANTILES = [0.1, 0.5, 0.9]
DST_THRESHOLD = -20.0
ZOOMED_IN_DAYS = 3

TIMESTAMP_COLS = ["timestamp", "time_tag", "date"]
DST_TARGET_CANDIDATES = ["h1", "dst", "Dst", "dst_value", "dst_dst"]


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _detect_ts(df: pd.DataFrame) -> str | None:
    for c in TIMESTAMP_COLS:
        if c in df.columns:
            return c
    return None


def _detect_dst(df: pd.DataFrame) -> str | None:
    for c in DST_TARGET_CANDIDATES:
        if c in df.columns:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else None


def _as_ts(val):
    ts = pd.to_datetime(val)
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts
