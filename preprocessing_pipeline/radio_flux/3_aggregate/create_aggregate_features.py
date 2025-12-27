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

from preprocessing_pipeline.utils import load_hourly_output, resample_to_hourly

STAGE_DIR = Path(__file__).resolve().parent

FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "radio_flux"
    / "2_engineered_features"
    / "radio_flux_filt_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "radio_flux_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"
MIN_ROWS_PER_WINDOW = 2   # radio flux is sparse; allow low coverage

# ---------------------------------------------------------------------
# Load hourly radio flux features
# ---------------------------------------------------------------------
def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("Radio flux engineered dataset is empty.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected radio flux indexed by timestamp.")
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    deltas = df.index.to_series().diff().dropna()
    if not deltas.empty and deltas.median() >= pd.Timedelta(hours=6):
        df = resample_to_hourly(df, method="ffill")
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
        f107 = window["f107"].dropna()
        if f107.empty:
            continue

        last_f107 = float(f107.iloc[-1])
        prev_f107 = f107.iloc[-2] if len(f107) >= 2 else np.nan

        # Log transform (safe)
        log_f107 = np.log(last_f107) if last_f107 > 0 else np.nan

        row = {
            "timestamp": window_end,

            # Current level
            "f107": last_f107,
            "log_f107": log_f107,

            # Short-term behavior (8h)
            "f107_mean_8h": float(f107.mean()),
            "f107_std_8h": float(f107.std(ddof=0)),
            "f107_delta_8h": (
                last_f107 - prev_f107
                if np.isfinite(prev_f107)
                else np.nan
            ),

            # Regime (absolute level)
            "f107_regime": int(
                0 if last_f107 < 80
                else 1 if last_f107 < 150
                else 2
            ),
        }

        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h radio flux features produced.")

    features = features.sort_values("timestamp").reset_index(drop=True)
    deltas = features["timestamp"].diff().dropna()
    if not deltas.empty and deltas.median() >= pd.Timedelta(hours=12):
        expanded = []
        for _, row in features.iterrows():
            base = row["timestamp"].floor("D")
            for offset in (0, 8, 16):
                new_row = row.copy()
                new_row["timestamp"] = base + pd.Timedelta(hours=offset)
                expanded.append(new_row)
        features = pd.DataFrame(expanded).sort_values("timestamp").reset_index(drop=True)
    return features


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_8h_radio_flux_features() -> pd.DataFrame:
    df = _load_features()
    features = _build_8h(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h radio flux features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_8h_radio_flux_features()


if __name__ == "__main__":
    main()
