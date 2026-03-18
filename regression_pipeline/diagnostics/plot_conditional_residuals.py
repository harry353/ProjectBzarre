from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

# === Configuration ===
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

# Directory containing serialised diagnostic polynomial models.
MODEL_DIR = PROJECT_ROOT / "regression_pipeline" / "diagnostics" / "polynomial_models"
# Forecast horizons h1 through h8.
HORIZONS = range(1, 9)
# Splits to generate plots for; "all" concatenates train + validation + test.
SPLITS = ("all", "train", "validation", "test")
# Polynomial degree and Ridge alpha matching those used in train_polynomial_regression.py.
DEGREE = 2
ALPHA = 1.0

NUM_BINS = 6  # quantile bins for y_pred
# Root output directory; one subdirectory is created per split.
OUTPUT_BASE = PROJECT_ROOT / "regression_pipeline" / "diagnostics" / "conditional_residual_plots"


# === Data / model loading ===
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


def _prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    # Drop non-numeric columns and rows with any NaN, then separate y and X.
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if target_col not in numeric.columns:
        raise RuntimeError(f"Missing target column '{target_col}' after numeric filtering.")
    y = numeric[target_col].astype(float).to_numpy()
    X = numeric.drop(columns=[target_col])
    if X.empty:
        raise RuntimeError("No numeric features available after filtering.")
    return y, X.to_numpy(dtype=np.float32)


def _model_path(horizon: int, degree: int, alpha: float) -> Path:
    # Build the filename that matches the serialisation convention in train_polynomial_regression.py.
    alpha_token = str(alpha).replace(".", "p")
    return MODEL_DIR / f"poly_h{horizon}_deg{degree}_alpha{alpha_token}.joblib"


def load_data_from_db(horizon: int, split: str) -> Tuple[np.ndarray, np.ndarray]:
    # Validate required paths before loading anything.
    if not FEATURES_DB.exists():
        raise FileNotFoundError(f"Features DB not found: {FEATURES_DB}")
    if not LABELS_DB.exists():
        raise FileNotFoundError(f"Labels DB not found: {LABELS_DB}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    model_path = _model_path(horizon, DEGREE, ALPHA)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load(model_path)

    target_col = f"h{horizon}"
    # "all" concatenates all three splits so the full dataset is visualised together.
    if split == "all":
        frames = [_merge_split(s, target_col) for s in ("train", "validation", "test")]
        split_df = pd.concat(frames, ignore_index=True)
    else:
        split_df = _merge_split(split, target_col)
    y_true, X = _prepare_xy(split_df, target_col)
    y_pred = model.predict(X).astype(float)
    return y_true, y_pred


# === Plotting ===
def make_quantile_bins(y_pred: np.ndarray, num_bins: int) -> np.ndarray:
    """Return bin edges (num_bins+1) using quantiles; ensures strictly increasing edges."""
    probs = np.linspace(0, 1, num_bins + 1)
    edges = np.quantile(y_pred, probs)
    # Nudge duplicate edges so all bins are non-degenerate.
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def plot_conditional_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, num_bins: int, output_path: Path, horizon: int, split: str
) -> None:
    # Compute residuals and drop any non-finite values before binning.
    residuals = np.asarray(y_true - y_pred, dtype=float).reshape(-1)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        raise RuntimeError("No finite residuals available to plot.")
    edges = make_quantile_bins(y_pred, num_bins)
    # digitize returns 1-based bin indices; subtract 1 to make them 0-based.
    bin_indices = np.digitize(y_pred, edges, right=False) - 1  # bins 0..num_bins-1
    # Clip to guard against edge values landing outside [0, num_bins-1].
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Share the x-axis range across all subplots so distributions are comparable.
    shared_xlim = (residuals.min(), residuals.max())
    rows = int(np.ceil(num_bins / 3))
    cols = int(np.ceil(num_bins / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for b in range(num_bins):
        ax = axes[b]
        mask = bin_indices == b
        r_bin = residuals[mask]
        lower = edges[b]
        upper = edges[b + 1]
        # Manual histogram to avoid numpy broadcast issues
        r_min, r_max = shared_xlim
        # Ensure the bin range is non-degenerate for linspace.
        if r_min == r_max:
            r_min -= 0.5
            r_max += 0.5
        bins = np.linspace(r_min, r_max, 41)
        counts = np.zeros(len(bins) - 1, dtype=float)
        # Use searchsorted for manual binning to avoid numpy broadcast issues.
        idx = np.searchsorted(bins, r_bin, side="right") - 1
        in_range = (idx >= 0) & (idx < counts.size)
        idx = idx[in_range]
        np.add.at(counts, idx, 1)
        widths = np.diff(bins)
        # Normalise counts to a density; floor at 1e-12 to support log-scale y-axis.
        density = counts / (counts.sum() * widths) if counts.sum() > 0 else counts
        density = np.maximum(density, 1e-12)
        ax.bar(bins[:-1], density, width=widths, align="edge", color="#4c72b0", alpha=0.75)
        # Red dashed line marks zero residual (perfect prediction).
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlim(shared_xlim)
        ax.set_title(f"ŷ in [{lower:.3g}, {upper:.3g})\nN={r_bin.size}")
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide any unused subplot axes when num_bins doesn't fill the grid evenly.
    for ax in axes[num_bins:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Conditional residual distributions by predicted value bins (h{horizon}, split={split})",
        fontsize=12,
    )
    fig.supxlabel("Residual (y - ŷ)")
    fig.supylabel("Density (log scale)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print bin edges to stdout for quick inspection of the quantile split points.
    print("Bin edges (ŷ quantiles):")
    for i in range(len(edges) - 1):
        print(f"  Bin {i}: [{edges[i]:.6g}, {edges[i+1]:.6g})")
    print(f"Saved plot to {output_path}")


def main() -> None:
    # Iterate over every split × horizon combination and save one PNG per pair.
    for split in SPLITS:
        split_dir = OUTPUT_BASE / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for horizon in HORIZONS:
            y_true, y_pred = load_data_from_db(horizon, split)
            if y_true.shape != y_pred.shape:
                raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            output_path = split_dir / f"conditional_residuals_h{horizon}_{split}.png"
            plot_conditional_residuals(y_true, y_pred, NUM_BINS, output_path, horizon, split)


if __name__ == "__main__":
    main()
