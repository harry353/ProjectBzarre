from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_DIR = Path(__file__).resolve().parent

FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "dst"
    / "5_engineered_features"
    / "dst_aver_filt_imp_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "dst_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"
MIN_ROWS_PER_WINDOW = 4   # ≥4 of 8 hours required

# ---------------------------------------------------------------------
# Load hourly DST engineered data
# ---------------------------------------------------------------------
def _load_hourly() -> pd.DataFrame:
    if not FEATURES_DB.exists():
        raise FileNotFoundError(f"DST engineered DB missing: {FEATURES_DB}")

    with sqlite3.connect(FEATURES_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {FEATURES_TABLE}",
            conn,
            parse_dates=["time_tag"],
        )

    if df.empty:
        raise RuntimeError("DST engineered dataset is empty.")

    df = df.dropna(subset=["time_tag"])
    return df.set_index("time_tag").sort_index()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _linear_slope(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan

    y = values.to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]

    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)

    if denom == 0:
        return 0.0

    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


# ---------------------------------------------------------------------
# Build 8h DST features (PAST WINDOW ONLY)
# ---------------------------------------------------------------------
def create_8h_features() -> pd.DataFrame:
    hourly = _load_hourly()

    rows: List[Dict[str, float]] = []

    grouped = hourly.groupby(
        pd.Grouper(freq=AGG_FREQ, label="right", closed="right")
    )

    for window_end, window in grouped:
        if window.empty or len(window) < MIN_ROWS_PER_WINDOW:
            continue

        dst_series = window["dst"].dropna()
        if dst_series.empty:
            continue

        record: Dict[str, float] = {}
        record["timestamp"] = window_end

        # Current state
        record["dst"] = float(dst_series.iloc[-1])

        # Extremes and variability (past 8h)
        record["dst_min_8h"] = float(dst_series.min())
        record["dst_mean_8h"] = float(dst_series.mean())
        record["dst_std_8h"] = float(dst_series.std(ddof=0))

        # Change over window
        record["dst_delta_8h"] = float(dst_series.iloc[-1] - dst_series.iloc[0])
        record["dst_slope_8h"] = _linear_slope(dst_series)

        # Recent derivative
        derivative = window["dst_derivative"].dropna()
        record["dst_derivative"] = (
            float(derivative.iloc[-1]) if not derivative.empty else 0.0
        )

        # Recovery indicator (causal)
        record["dst_recovery_flag"] = int(
            record["dst"] < 0 and record["dst_derivative"] > 0
        )

        rows.append(record)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h DST features produced.")

    features.sort_values("timestamp", inplace=True)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h DST features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    create_8h_features()


if __name__ == "__main__":
    main()
