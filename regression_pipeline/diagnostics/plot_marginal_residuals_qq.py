from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from scipy import stats


# Resolve project root two levels above this diagnostics script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# PCA-projected feature tables produced by apply_pca_transforms.py.
FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
# Dst regression label tables — one column per forecast horizon.
LABELS_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "labels"
    / "dst_regression"
    / "dst_regression_labels.db"
)

# Table names for each split in the features DB.
FEATURE_TABLES = {
    "train": "pca_train",
    "validation": "pca_validation",
    "test": "pca_test",
}
# Table names for each split in the labels DB.
LABEL_TABLES = {
    "train": "dst_regression_train",
    "validation": "dst_regression_validation",
    "test": "dst_regression_test",
}

# Forecast horizons h1 through h8.
HORIZONS = range(1, 9)
# Directory containing serialised diagnostic polynomial models.
MODEL_DIR = PROJECT_ROOT / "regression_pipeline" / "diagnostics" / "polynomial_models"
# Output directory for the residual distribution and Q-Q plot PNGs.
PLOTS_DIR = PROJECT_ROOT / "regression_pipeline" / "diagnostics" / "marginal_residual_dist_and_qq_plot"
# Polynomial degree and Ridge alpha matching those used in train_polynomial_regression.py.
DEFAULT_DEGREE = 2
DEFAULT_ALPHA = 1.0
# Split to evaluate; only test split is used to avoid evaluating on seen data.
DEFAULT_SPLIT = "test"


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    # Read the full table from SQLite into a DataFrame.
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    # Parse timestamps to UTC-aware then strip timezone for consistent merging.
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts.dt.tz_localize(None)


def _merge_split(split: str, target_col: str) -> pd.DataFrame:
    # Inner-join features and the single target-horizon label column on timestamp.
    features = _load_table(FEATURES_DB, FEATURE_TABLES[split])
    labels = _load_table(LABELS_DB, LABEL_TABLES[split])[["timestamp", target_col]]
    features["timestamp"] = _normalize_timestamp(features["timestamp"])
    labels["timestamp"] = _normalize_timestamp(labels["timestamp"])
    merged = features.merge(labels, on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError(f"No merged rows for split '{split}'.")
    return merged


def _prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    # Drop non-numeric columns and rows with any NaN, then separate X and y.
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if target_col not in numeric.columns:
        raise RuntimeError(f"Missing target column '{target_col}' after numeric filtering.")
    y = numeric[target_col].astype(float).to_numpy()
    X = numeric.drop(columns=[target_col])
    if X.empty:
        raise RuntimeError("No numeric features available after filtering.")
    return X.to_numpy(dtype=np.float32), y


def _model_path(horizon: int, degree: int, alpha: float) -> Path:
    # Build the filename that matches the serialisation convention in train_polynomial_regression.py.
    alpha_token = str(alpha).replace(".", "p")
    return MODEL_DIR / f"poly_h{horizon}_deg{degree}_alpha{alpha_token}.joblib"


def _plot_residuals(residuals: np.ndarray, horizon: int, split: str) -> plt.Figure:
    # Flatten and coerce residuals to a clean 1-D float array, dropping non-finite values.
    try:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        residuals = np.concatenate(
            [np.ravel(np.asarray(part, dtype=float)) for part in residuals]
        )
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise RuntimeError("No finite residuals available to plot.")
    # Three-panel figure: raw residual histogram | standardised residual histogram | Q-Q plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Manual histogram to avoid numpy/matplotlib bin broadcast quirks.
    r_min, r_max = residuals.min(), residuals.max()
    # Ensure the bin range is non-degenerate for linspace.
    if r_min == r_max:
        r_min -= 0.5
        r_max += 0.5
    bin_edges = np.linspace(r_min, r_max, 61)
    counts = np.zeros(len(bin_edges) - 1, dtype=float)
    # Manual binning using searchsorted to avoid numpy broadcast issues.
    bin_idx = np.searchsorted(bin_edges, residuals, side="right") - 1
    in_range = (bin_idx >= 0) & (bin_idx < counts.size)
    bin_idx = bin_idx[in_range]
    np.add.at(counts, bin_idx, 1)
    widths = np.diff(bin_edges)
    density = counts / (counts.sum() * widths)
    density = np.maximum(density, 1e-12)  # avoid zeros for log-scale
    axes[0].bar(
        bin_edges[:-1],
        density,
        width=widths,
        align="edge",
        color="#4c72b0",
        alpha=0.75,
    )
    axes[0].set_title(f"Residuals (h{horizon}, {split})")
    axes[0].set_xlabel("Residual (y - y_hat)")
    axes[0].set_ylabel("Density")
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Standardized residuals using global sample std as scale proxy
    sigma_hat = np.std(residuals, ddof=1)
    # Guard against zero std — substitute 1.0 so z = residuals (no scaling).
    if sigma_hat <= 0:
        sigma_hat = 1.0
    z = residuals / sigma_hat
    z_min, z_max = z.min(), z.max()
    if z_min == z_max:
        z_min -= 0.5
        z_max += 0.5
    z_edges = np.linspace(z_min, z_max, 61)
    z_counts = np.zeros(len(z_edges) - 1, dtype=float)
    z_idx = np.searchsorted(z_edges, z, side="right") - 1
    in_range = (z_idx >= 0) & (z_idx < z_counts.size)
    z_idx = z_idx[in_range]
    np.add.at(z_counts, z_idx, 1)
    z_widths = np.diff(z_edges)
    z_density = z_counts / (z_counts.sum() * z_widths)
    # Floor at 1e-12 so log-scale y-axis renders without -inf bars.
    z_density = np.maximum(z_density, 1e-12)
    axes[1].bar(
        z_edges[:-1],
        z_density,
        width=z_widths,
        align="edge",
        color="#4c72b0",
        alpha=0.75,
    )
    # Red dashed line marks z = 0 (perfect prediction).
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title(f"Standardized residuals z (h{horizon}, {split})")
    axes[1].set_xlabel("z = (y - ŷ) / σ̂")
    axes[1].set_ylabel("Density")
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Normal Q-Q plot — deviations from the diagonal indicate non-Gaussian tails.
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title(f"QQ Plot (h{horizon}, {split})")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


def main() -> None:
    # Pre-flight validation before any data is loaded.
    if DEFAULT_SPLIT not in ("train", "validation", "test"):
        raise ValueError(f"Unsupported split: {DEFAULT_SPLIT}")
    if not FEATURES_DB.exists():
        raise FileNotFoundError(f"Features DB not found: {FEATURES_DB}")
    if not LABELS_DB.exists():
        raise FileNotFoundError(f"Labels DB not found: {LABELS_DB}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        model_path = _model_path(horizon, DEFAULT_DEGREE, DEFAULT_ALPHA)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = load(model_path)
        target_col = f"h{horizon}"
        split_df = _merge_split(DEFAULT_SPLIT, target_col)
        X_split, y_split = _prepare_xy(split_df, target_col)
        preds = model.predict(X_split)
        # Flatten to 1-D in case the model returns a 2-D array.
        residuals = np.ravel(y_split - preds)
        fig = _plot_residuals(residuals, horizon, DEFAULT_SPLIT)
        out_path = PLOTS_DIR / f"residuals_h{horizon}_{DEFAULT_SPLIT}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
