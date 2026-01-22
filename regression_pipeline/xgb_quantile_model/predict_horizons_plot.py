from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
LABELS_DB = PROJECT_ROOT / "preprocessing_pipeline" / "labels" / "dst_regression" / "dst_regression_labels.db"

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

MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_model"
HORIZONS = range(1, 2)
QUANTILES = [0.1, 0.5, 0.9]
YEAR = 2024  # edit as needed
LAST_DAYS = 10


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("UTC")
    return ts.dt.tz_localize(None)


def _load_full_split(split: str) -> pd.DataFrame:
    features = _load_table(FEATURES_DB, FEATURE_TABLES[split])
    labels = _load_table(LABELS_DB, LABEL_TABLES[split])
    feature_cols = [
        c for c in features.columns
        if c != "timestamp" and np.issubdtype(features[c].dtype, np.number)
    ]
    features["timestamp"] = _normalize_timestamp(features["timestamp"])
    labels["timestamp"] = _normalize_timestamp(labels["timestamp"])
    merged = features.merge(labels, on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError(f"No merged rows for split '{split}'.")
    return merged, feature_cols


def load_year(year: int) -> pd.DataFrame:
    frames = []
    feature_cols = None
    for split in ("train", "validation", "test"):
        df, fcols = _load_full_split(split)
        feature_cols = feature_cols or fcols
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all["timestamp"].dt.year == year].copy()
    if df_all.empty:
        raise RuntimeError(f"No rows found for year {year}.")
    df_all = df_all.dropna(axis=0, how="any")
    return df_all, feature_cols


def plot_year(year: int) -> None:
    if not FEATURES_DB.exists():
        raise FileNotFoundError(f"Features DB not found: {FEATURES_DB}")
    if not LABELS_DB.exists():
        raise FileNotFoundError(f"Labels DB not found: {LABELS_DB}")
    if not MODEL_BASE.exists():
        raise FileNotFoundError(f"Model base dir not found: {MODEL_BASE}")

    df_all, feature_cols = load_year(year)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_all["timestamp"], df_all["h1"], label="DST (actual)", color="#1f77b4")

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    rmses = []
    for idx, h in enumerate(HORIZONS):
        target_col = f"h{h}"
        numeric = df_all[feature_cols + [target_col]].dropna(axis=0, how="any")
        if target_col not in numeric.columns:
            raise RuntimeError(f"Missing target column '{target_col}' after numeric filtering.")
        X = numeric[feature_cols].to_numpy(dtype=np.float32)
        ts = df_all.loc[numeric.index, "timestamp"]

        preds = {}
        for q in QUANTILES:
            model_path = MODEL_BASE / f"h{h}" / f"q{q}" / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Quantile model not found: {model_path}")
            model = joblib.load(model_path)
            preds[q] = model.predict(X).astype(float)

        # RMSE for this horizon over all available rows
        y_true = numeric[target_col].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((y_true - preds[0.5]) ** 2)))
        rmses.append(rmse)
        print(f"[RMSE] h{h} (median): {rmse:.4f} (n={len(y_true)})")

        ts_shifted = ts + pd.Timedelta(hours=0)
        color = colors[idx % len(colors)]
        ax.fill_between(
            ts_shifted,
            preds[0.1],
            preds[0.9],
            color=color,
            alpha=0.2,
            label=f"h{h} q0.1–q0.9",
        )
        ax.plot(ts_shifted, preds[0.5], label=f"h{h} q0.5", color=color, linestyle="--", linewidth=1.6)

    ax.set_title(f"DST vs forecasts (aligned) - {year}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("DST")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    # ax.set_ylim(-50, 20)
    if not df_all.empty:
        year_end = pd.Timestamp(year=year, month=12, day=31, tz='UTC')
        x_max = df_all["timestamp"].max()
        if pd.isna(x_max):
            x_max = year_end
        x_min = x_max - pd.Timedelta(days=LAST_DAYS)
        # ax.set_xlim(left=x_min, right=x_max + pd.Timedelta(hours=6))
    fig.autofmt_xdate()
    plt.show()

    if rmses:
        avg_rmse = float(np.mean(rmses))
        print(f"[RMSE] Average across horizons: {avg_rmse:.4f}")


if __name__ == "__main__":
    plot_year(YEAR)
