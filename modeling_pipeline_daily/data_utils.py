from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_split_tables(merged_db: Path) -> Dict[str, pd.DataFrame]:
    if not merged_db.exists():
        raise FileNotFoundError(f"Merged dataset not found at {merged_db}")

    splits: Dict[str, pd.DataFrame] = {}
    with sqlite3.connect(merged_db) as conn:
        for split in ("train", "validation", "test"):
            table = f"merged_{split}"
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if df.empty:
                raise RuntimeError(f"No rows found in {table} of {merged_db}")

            time_col = None
            for candidate in ("date", "timestamp"):
                if candidate in df.columns:
                    time_col = candidate
                    break

            label_col = None
            if "storm_labels_storm_next_24h" in df.columns:
                label_col = "storm_labels_storm_next_24h"
            elif "storm_labels_storm_severity_next_8h" in df.columns:
                label_col = "storm_labels_storm_severity_next_8h"
            else:
                for col in df.columns:
                    if col.startswith("storm_labels_"):
                        label_col = col
                        break

            if time_col is None or label_col is None:
                raise RuntimeError(
                    f"Required label columns missing in {table}. Expected date/timestamp and "
                    "storm label column."
                )

            timestamps = pd.to_datetime(df[time_col], errors="coerce")
            if timestamps.dt.tz is None:
                timestamps = timestamps.dt.tz_localize("UTC")
            else:
                timestamps = timestamps.dt.tz_convert("UTC")
            df["forecast_date"] = timestamps

            label_series = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
            df["storm_present_next_24h"] = (label_series > 0).astype(int)

            drop_cols = [
                time_col,
                label_col,
            ]
            drop_cols += [c for c in df.columns if c.endswith("_timestamp") or c.endswith("_time_tag")]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

            splits[split] = df
    return splits


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {"forecast_date", "storm_present_next_24h"}
    return [c for c in df.columns if c not in drop_cols and not c.startswith("storm_labels_")]


def prepare_arrays(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = df["storm_present_next_24h"].to_numpy(dtype=np.int8)
    return X, y


def compute_sample_weights(
    train_df: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float]]:
    weights = np.ones(len(train_df), dtype=np.float32)
    stats = {
        "definition": "Uniform weights",
        "alpha": 0.0,
        "mean_positive_severity": 0.0,
        "mean_positive_weight": 1.0,
        "max_weight": 1.0,
    }
    return weights, stats
