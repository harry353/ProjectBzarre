from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
LABELS_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "labels"
    / "dst_regression"
    / "dst_regression_labels.db"
)

FEATURE_TABLES = {
    "train": "pca_train",
    "validation": "pca_validation",
    "test": "pca_test",
}
LABEL_TABLES = {
    "train": "dst_regression_train",
    "validation": "dst_regression_validation",
    "test": "dst_regression_test",
}

HORIZONS = range(1, 9)
DEFAULT_DEGREE = 2
DEFAULT_ALPHA = 1.0
MODEL_DIR = PROJECT_ROOT / "regression_pipeline" / "polynomial_models"


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts.dt.tz_localize(None)


def _merge_split(split: str, target_col: str) -> pd.DataFrame:
    features = _load_table(FEATURES_DB, FEATURE_TABLES[split])
    labels = _load_table(LABELS_DB, LABEL_TABLES[split])[["timestamp", target_col]]
    features["timestamp"] = _normalize_timestamp(features["timestamp"])
    labels["timestamp"] = _normalize_timestamp(labels["timestamp"])
    merged = features.merge(labels, on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError(f"No merged rows for split '{split}'.")
    return merged


def _prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if target_col not in numeric.columns:
        raise RuntimeError(f"Missing target column '{target_col}' after numeric filtering.")
    y = numeric[target_col].astype(float).to_numpy()
    X = numeric.drop(columns=[target_col])
    if X.empty:
        raise RuntimeError("No numeric features available after filtering.")
    return X.to_numpy(dtype=np.float32), y, X.columns.tolist()


def _fit_and_report(horizon: int, degree: int, alpha: float):
    target_col = f"h{horizon}"
    train_df = _merge_split("train", target_col)
    val_df = _merge_split("validation", target_col)
    test_df = _merge_split("test", target_col)

    X_train, y_train, _ = _prepare_xy(train_df, target_col)
    X_val, y_val, _ = _prepare_xy(val_df, target_col)
    X_test, y_test, _ = _prepare_xy(test_df, target_col)

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=alpha, random_state=42),
    )
    model.fit(X_train, y_train)

    def _metrics(split: str, X: np.ndarray, y: np.ndarray) -> None:
        pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        print(f"  {split}: RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

    print(f"[h{horizon}] degree={degree} alpha={alpha}")
    _metrics("train", X_train, y_train)
    _metrics("val", X_val, y_val)
    _metrics("test", X_test, y_test)
    return model


def _model_path(horizon: int, degree: int, alpha: float) -> Path:
    alpha_token = str(alpha).replace(".", "p")
    return MODEL_DIR / f"poly_h{horizon}_deg{degree}_alpha{alpha_token}.joblib"


def main() -> None:
    if not FEATURES_DB.exists():
        raise FileNotFoundError(f"Features DB not found: {FEATURES_DB}")
    if not LABELS_DB.exists():
        raise FileNotFoundError(f"Labels DB not found: {LABELS_DB}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trained = []
    for h in HORIZONS:
        if h not in HORIZONS:
            raise ValueError(f"Unsupported horizon: {h}")
        model = _fit_and_report(h, DEFAULT_DEGREE, DEFAULT_ALPHA)
        trained.append((h, model))
    for h, model in trained:
        model_path = _model_path(h, DEFAULT_DEGREE, DEFAULT_ALPHA)
        dump(model, model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
