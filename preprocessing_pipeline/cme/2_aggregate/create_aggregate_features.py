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

HOURLY_DB = (
    STAGE_DIR.parents[1]
    / "cme"
    / "1_engineered_features"
    / "cme_hourly_eng.db"
)
HOURLY_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "cme_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
AGG_FREQ = "8H"                      # 8-hour cadence
MIN_ROWS_PER_WINDOW = 4              # require ≥4 of 8 hours

# ---------------------------------------------------------------------
# Load hourly CME data
# ---------------------------------------------------------------------
def _load_hourly() -> pd.DataFrame:
    if not HOURLY_DB.exists():
        raise FileNotFoundError(f"Missing CME hourly DB: {HOURLY_DB}")

    with sqlite3.connect(HOURLY_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {HOURLY_TABLE}",
            conn,
            parse_dates=["time_tag"],
        )

    if df.empty:
        raise RuntimeError("CME hourly dataset is empty.")

    df = df.dropna(subset=["time_tag"])
    return df.set_index("time_tag").sort_index()


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------
def _max_flag(window: pd.DataFrame, col: str) -> int:
    if col not in window:
        return 0
    return int((window[col] > 0).any())


def _last_value(window: pd.DataFrame, col: str) -> float:
    if col not in window:
        return np.nan
    data = window[col].dropna()
    return float(data.iloc[-1]) if not data.empty else np.nan


def _sum(window: pd.DataFrame, col: str) -> float:
    if col not in window:
        return np.nan
    return float(window[col].sum(skipna=True))


# ---------------------------------------------------------------------
# Build 8h CME features (PAST WINDOW ONLY)
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

        record: Dict[str, float] = {}
        record["timestamp"] = window_end

        # Event flags (past window)
        record["cme_active_flag"] = _max_flag(window, "cme_active_flag")
        record["cme_overtaking_flag"] = _max_flag(window, "cme_overtaking_flag")
        record["earth_facing_flag"] = _max_flag(window, "earth_facing_flag")
        record["last_cme_fast_flag"] = _max_flag(window, "last_cme_fast_flag")

        # Counts / integrated activity
        record["cme_count_8h"] = _sum(window, "cme_count_last_24h")
        record["cme_strength_sum_8h"] = _sum(window, "cme_strength_sum_24h")

        # Last-known CME state (causal)
        record["hours_since_last_cme"] = _last_value(window, "hours_since_last_cme")
        record["last_cme_v_med"] = _last_value(window, "last_cme_v_med")
        record["last_cme_width"] = _last_value(window, "last_cme_width")
        record["last_cme_strength"] = _last_value(window, "last_cme_strength")
        record["last_cme_shock_proxy"] = _last_value(window, "last_cme_shock_proxy")
        record["last_cme_speed_ratio"] = _last_value(window, "last_cme_speed_ratio")
        record["delta_last_cme_speed"] = _last_value(window, "delta_last_cme_speed")
        record["delta_last_cme_width"] = _last_value(window, "delta_last_cme_width")
        record["earth_alignment_score"] = _last_value(window, "earth_alignment_score")
        record["effective_width"] = _last_value(window, "effective_width")
        record["cme_influence_exp"] = _last_value(window, "cme_influence_exp")
        record["cme_severity_class"] = _last_value(window, "cme_severity_class")

        rows.append(record)

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("No 8h CME features produced.")

    features.sort_values("timestamp", inplace=True)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] 8h CME features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    create_8h_features()


if __name__ == "__main__":
    main()
