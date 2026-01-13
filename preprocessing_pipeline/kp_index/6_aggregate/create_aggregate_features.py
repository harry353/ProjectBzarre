from __future__ import annotations

import sys
from pathlib import Path
import sqlite3

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

from preprocessing_pipeline.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent

FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "kp_index"
    / "5_engineered_features"
    / "kp_index_aver_filt_imp_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "kp_index_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Aggregation parameters (HOURLY)
# ---------------------------------------------------------------------
WINDOW_H = 6
MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Load hourly KP features
# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("KP engineered dataset is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected KP features indexed by timestamp.")
    df = df.sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _linear_slope(series: pd.Series) -> float:
    y = series.to_numpy(dtype=float)
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
# Build HOURLY KP aggregate features
# ---------------------------------------------------------------------
def create_kp_agg_features() -> pd.DataFrame:
    df = _load_features()
    out = df.copy()

    min_periods = max(1, int(np.ceil(WINDOW_H * MIN_FRACTION_COVERAGE)))
    window = f"{WINDOW_H}h"

    # --------------------------------------------------------------
    # Aggregate features (5 total)
    # --------------------------------------------------------------
    out[f"kp_max_{WINDOW_H}h"] = (
        df["kp"]
        .rolling(window, min_periods=min_periods)
        .max()
    )

    out[f"kp_mean_{WINDOW_H}h"] = (
        df["kp"]
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out[f"kp_delta_{WINDOW_H}h"] = (
        df["kp"] - df["kp"].shift(WINDOW_H)
    )

    out[f"kp_ge5_frac_{WINDOW_H}h"] = (
        (df["kp"] >= 5.0)
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out[f"kp_slope_{WINDOW_H}h"] = (
        df["kp"]
        .rolling(window, min_periods=min_periods)
        .apply(_linear_slope, raw=False)
    )

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------
    out = out.dropna()
    if out.empty:
        raise RuntimeError("No KP aggregate features produced.")

    out = out.reset_index().rename(columns={"index": "timestamp"})

    with sqlite3.connect(OUTPUT_DB) as conn:
        out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] KP aggregate features written to {OUTPUT_DB}")
    print(f"Rows written: {len(out):,}")

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    create_kp_agg_features()


if __name__ == "__main__":
    main()
