from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TARGET_HORIZON_H = 4
TARGET_COL = f"storm_present_next_{TARGET_HORIZON_H}h"


def _target_col(target_horizon_h: int | None) -> str:
    horizon = TARGET_HORIZON_H if target_horizon_h is None else int(target_horizon_h)
    return f"storm_present_next_{horizon}h"


def load_split_tables(
    merged_db: Path,
    target_horizon_h: int | None = None,
    label_source: str = "full_storm",
) -> Dict[str, pd.DataFrame]:
    if not merged_db.exists():
        raise FileNotFoundError(f"Merged dataset not found at {merged_db}")

    target_col = _target_col(target_horizon_h)
    label_prefix = f"{label_source}_labels_"
    desired_label = f"{label_prefix}storm_flag_h{target_horizon_h or TARGET_HORIZON_H}"

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
            if desired_label in df.columns:
                label_col = desired_label
            elif "storm_labels_storm_next_24h" in df.columns:
                label_col = "storm_labels_storm_next_24h"
            elif "storm_labels_storm_severity_next_8h" in df.columns:
                label_col = "storm_labels_storm_severity_next_8h"
            else:
                for col in df.columns:
                    if col.startswith(label_prefix) or col.startswith("storm_labels_"):
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
            df[target_col] = (label_series > 0).astype(int)

            drop_cols = [
                time_col,
                label_col,
            ]
            drop_cols += [c for c in df.columns if c.endswith("_timestamp") or c.endswith("_time_tag")]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

            splits[split] = df
    return splits


def feature_columns(df: pd.DataFrame, target_col: str | None = None) -> list[str]:
    drop_cols = {"forecast_date", target_col or TARGET_COL}
    return [
        c
        for c in df.columns
        if c not in drop_cols
        and not c.startswith("storm_labels_")
        and not c.startswith("full_storm_labels_")
        and not c.startswith("main_phase_labels_")
    ]


def prepare_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = df[target_col or TARGET_COL].to_numpy(dtype=np.int8)
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
