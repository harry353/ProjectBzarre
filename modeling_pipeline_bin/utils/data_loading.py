from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PIPELINE_ROOT / "data"
DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_HORIZON_HOURS = 4


def _load_parquet_directory(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise ValueError(f"No parquet files available in {directory}")
    df = pd.concat(frames, ignore_index=True)
    for time_col in ("timestamp", "time_tag"):
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)
            break
    return df


def _prepare_features_labels(df: pd.DataFrame, horizon_hours: int) -> Tuple[pd.DataFrame, pd.Series]:
    if "severity_label" not in df.columns:
        raise RuntimeError("Dataset must contain a 'severity_label' column.")

    label = df["severity_label"].shift(-horizon_hours)
    valid_mask = label.notna()
    if not valid_mask.any():
        raise ValueError(f"No rows remain after applying a {horizon_hours}h label shift.")

    # Drop rows where any feature or label is NaN to keep XGBoost happy.
    feature_cols = [col for col in df.columns if "label" not in col.lower()]
    subset_df = df.loc[valid_mask, feature_cols].copy()
    subset_df["severity_label"] = label.loc[valid_mask]
    subset_df = subset_df.dropna()

    y = subset_df.pop("severity_label").astype("Int8")
    X = subset_df
    return X, y


def load_split_dataframe(split: str, horizon_hours: int = DEFAULT_HORIZON_HOURS) -> Tuple[pd.DataFrame, pd.Series]:
    directory = DATA_DIR / split
    df = _load_parquet_directory(directory)
    return _prepare_features_labels(df, horizon_hours)


def load_split_arrays(split: str, horizon_hours: int = DEFAULT_HORIZON_HOURS) -> Tuple[np.ndarray, np.ndarray]:
    X_df, y = load_split_dataframe(split, horizon_hours)
    X = X_df.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy(dtype=np.int8)
    return X, y_arr


def load_all_splits(horizon_hours: int = DEFAULT_HORIZON_HOURS, splits = DEFAULT_SPLITS) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    data = {}
    for split in splits:
        data[split] = load_split_arrays(split, horizon_hours)
    return data


def main():
    split = DEFAULT_SPLITS[0]
    horizon = DEFAULT_HORIZON_HOURS
    directory = DATA_DIR / split
    raw_df = _load_parquet_directory(directory)
    features_df, labels = _prepare_features_labels(raw_df, horizon)

    # Sanity check: ensure shifted labels align with original column
    shifted = raw_df["severity_label"].shift(-horizon)
    nan_count = shifted.isna().sum()
    print(f"NaNs introduced by shifting: {nan_count}")
    if nan_count != horizon:
        raise AssertionError(
            f"Unexpected NaN count ({nan_count}) after shifting. "
            f"Expected {horizon} for a {horizon}h horizon."
        )
    expected = shifted.loc[labels.index].astype("Int8")
    if not expected.equals(labels):
        raise AssertionError("Sanity check failed: shifted labels do not match expected values.")

    X = features_df.to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.int8)
    print(f"Split: {split}")
    print(f"Horizon: {horizon}h")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution (counts): {np.unique(y, return_counts=True)}")
    print("Sanity check passed: labels correctly shifted.")


if __name__ == "__main__":
    main()
