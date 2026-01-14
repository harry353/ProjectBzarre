from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

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
# Aggregation parameters (HOURLY, PAST-ONLY)
# ---------------------------------------------------------------------
WINDOW_H = 6
MIN_FRACTION_COVERAGE = 0.5

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
# Build HOURLY DST aggregate features
# ---------------------------------------------------------------------
def create_dst_agg_features() -> pd.DataFrame:
    df = _load_hourly()
    out = df.copy()

    min_periods = max(1, int(np.ceil(WINDOW_H * MIN_FRACTION_COVERAGE)))
    window = f"{WINDOW_H}h"

    # --------------------------------------------------------------
    # Aggregate features (5 total)
    # --------------------------------------------------------------
    out[f"dst_min_{WINDOW_H}h"] = (
        df["dst"]
        .rolling(window, min_periods=min_periods)
        .min()
    )

    out[f"dst_mean_{WINDOW_H}h"] = (
        df["dst"]
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out[f"dst_delta_{WINDOW_H}h"] = (
        df["dst"] - df["dst"].shift(WINDOW_H)
    )

    out[f"dst_slope_{WINDOW_H}h"] = (
        df["dst"]
        .rolling(window, min_periods=min_periods)
        .apply(_linear_slope, raw=False)
    )

    out[f"dst_neg_frac_{WINDOW_H}h"] = (
        (df["dst"] < 0)
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------
    out = out.dropna()
    if out.empty:
        raise RuntimeError("No DST aggregate features produced.")

    out = out.reset_index().rename(columns={"index": "timestamp"})

    with sqlite3.connect(OUTPUT_DB) as conn:
        out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] DST aggregate features written to {OUTPUT_DB}")
    print(f"Rows written: {len(out):,}")

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    create_dst_agg_features()


if __name__ == "__main__":
    main()
