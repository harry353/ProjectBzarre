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
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"
MIN_ROWS_PER_WINDOW = 4   # ≥50% coverage

# ---------------------------------------------------------------------
# Load hourly KP / Ap features
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
# Build 8h aggregates (PAST-ONLY)
# ---------------------------------------------------------------------
def _build_8h(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(
        pd.Grouper(freq=AGG_FREQ, label="right", closed="right")
    )

    for window_end, window in grouped:
        if len(window) < MIN_ROWS_PER_WINDOW:
            continue

        window = window.sort_index()
        last = window.iloc[-1]

        kp_series = window["kp_index"].dropna()
        if kp_series.empty:
            continue

        last_kp = float(kp_series.iloc[-1])
        prev_kp = kp_series.iloc[-2] if len(kp_series) >= 2 else np.nan

        kp_minus_3 = kp_series.iloc[-4] if len(kp_series) >= 4 else np.nan
        kp_minus_6 = kp_series.iloc[-7] if len(kp_series) >= 7 else np.nan

        kp_delta_6h = (
            last_kp - kp_minus_6
            if np.isfinite(kp_minus_6)
            else np.nan
        )

        kp_accel = (
            (last_kp - kp_minus_3) - (kp_minus_3 - kp_minus_6)
            if np.isfinite(kp_minus_3) and np.isfinite(kp_minus_6)
            else np.nan
        )

        jump_2plus = int((kp_series.diff() >= 2.0).any())

        row = {
            "timestamp": window_end,

            # Current state
            "kp_index": last_kp,
            "kp_regime": int(last.get("kp_regime", 0)),

            # Extremes & persistence
            "kp_max_8h": float(kp_series.max()),
            "kp_mean_6h": float(kp_series.tail(6).mean()),
            "kp_hours_above_5": int((kp_series >= 5).sum()),

            # Dynamics
            "kp_delta_6h": kp_delta_6h,
            "kp_accel": kp_accel,
            "kp_dist_to_5": last_kp - 5.0,

            # Activity flags
            "kp_jump_2plus": jump_2plus,
            "kp_entered_storm": int(last.get("kp_entered_storm", 0)),

            # Ap coupling (past-only)
            "ap_sum_8h": float(window["ap"].sum()),
            "ap_max_8h": float(window["ap"].max()),
        }

        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h KP features produced.")

    features = features.sort_values("timestamp").reset_index(drop=True)
    return features


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_8h_kp_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_8h(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h KP features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_8h_kp_features()


if __name__ == "__main__":
    main()

