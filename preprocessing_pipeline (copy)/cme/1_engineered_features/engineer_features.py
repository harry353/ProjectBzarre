from __future__ import annotations

import sys
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("Project root not found (space_weather_api.py missing)")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline.utils import write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

CATALOG_DB = STAGE_DIR.parents[1] / "space_weather.db"
CATALOG_TABLE = "lasco_cme_catalog"

OUTPUT_DB = STAGE_DIR / "cme_hourly_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------
START = pd.Timestamp("1998-01-01T00:00:00Z")
END   = pd.Timestamp("2025-11-30T23:00:00Z")
HOURLY_INDEX = pd.date_range(START, END, freq="1h", tz="UTC")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DECAY_TAU_HOURS = 36.0

# ---------------------------------------------------------------------
# Load CME catalog
# ---------------------------------------------------------------------
def _load_cme_catalog() -> pd.DataFrame:
    with sqlite3.connect(CATALOG_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {CATALOG_TABLE}",
            conn,
            parse_dates=["time_tag"],
        )
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    return df.sort_values("time_tag")

# ---------------------------------------------------------------------
# Feature engineering (HOURLY, REDUCED)
# ---------------------------------------------------------------------
def engineer_cme_features() -> pd.DataFrame:
    cme = _load_cme_catalog()
    cme = cme.set_index("time_tag").sort_index()
    cme = cme.dropna(subset=["median_velocity"])
    cme = cme.loc[~cme.index.duplicated(keep="last")]

    # --------------------------------------------------------------
    # CME interaction + strength proxies
    # --------------------------------------------------------------
    cme["prev_cme_v_med"] = cme["median_velocity"].shift(1)
    cme["last_cme_speed_ratio"] = (
        cme["median_velocity"] / cme["prev_cme_v_med"].clip(lower=1.0)
    )
    cme["cme_overtaking_flag"] = (cme["last_cme_speed_ratio"] > 1.5).astype(int)

    cme["strength"] = cme["median_velocity"] * cme["angular_width"]
    cme["shock_proxy"] = (
        cme["median_velocity"]
        * (cme["velocity_variation"] / cme["median_velocity"].clip(lower=1.0))
    )

    pa = np.deg2rad(cme["position_angle"])
    earth_alignment = np.cos(pa).clip(-1.0, 1.0)
    cme["effective_width"] = cme["angular_width"] * earth_alignment.clip(0.0, 1.0)

    # --------------------------------------------------------------
    # Hourly backbone
    # --------------------------------------------------------------
    hourly = pd.DataFrame(index=HOURLY_INDEX)
    hourly.index.name = "time_tag"

    # --------------------------------------------------------------
    # Time since last CME (monotonic-safe)
    # --------------------------------------------------------------
    event_index = (
        cme.index
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if not event_index.is_monotonic_increasing:
        raise RuntimeError("CME event index not monotonic")

    event_series = pd.Series(event_index, index=event_index)

    last_event = event_series.reindex(
        hourly.index,
        method="ffill",
    )

    hourly["hours_since_last_cme"] = (
        (hourly.index - last_event).dt.total_seconds() / 3600.0
    ).fillna(1e6)

    # --------------------------------------------------------------
    # CME rate / strength
    # --------------------------------------------------------------
    strength_series = (
        cme["strength"]
        .groupby(pd.Grouper(freq="1h"))
        .sum()
        .reindex(hourly.index, fill_value=0.0)
    )

    hourly["cme_strength_sum_24h"] = (
        strength_series.rolling(24, min_periods=1).sum()
    )

    # --------------------------------------------------------------
    # Carry-forward CME properties (FIXED)
    # --------------------------------------------------------------
    def ff(series: pd.Series, fill=0.0) -> pd.Series:
        s = (
            series
            .dropna()
            .loc[~series.index.duplicated(keep="last")]
            .sort_index()
        )

        # ALIGN first, THEN forward-fill
        aligned = s.reindex(hourly.index)
        return aligned.ffill().fillna(fill)


    hourly["last_cme_v_med"] = ff(cme["median_velocity"])
    hourly["effective_width"] = ff(cme["effective_width"])
    hourly["cme_overtaking_flag"] = ff(cme["cme_overtaking_flag"], 0).astype(int)
    hourly["last_cme_shock_proxy"] = ff(cme["shock_proxy"])

    # --------------------------------------------------------------
    # Influence decay
    # --------------------------------------------------------------
    hourly["cme_influence_exp"] = np.exp(
        -hourly["hours_since_last_cme"] / DECAY_TAU_HOURS
    )

    # --------------------------------------------------------------
    # Final column selection (ONLY 7)
    # --------------------------------------------------------------
    hourly = hourly[
        [
            "hours_since_last_cme",
            "last_cme_v_med",
            "effective_width",
            "cme_strength_sum_24h",
            "cme_overtaking_flag",
            "cme_influence_exp",
            "last_cme_shock_proxy",
        ]
    ]

    write_sqlite_table(hourly, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] CME engineered features written to {OUTPUT_DB}")
    print(f"Rows written: {len(hourly):,}")

    return hourly

# ---------------------------------------------------------------------
def main() -> None:
    engineer_cme_features()

if __name__ == "__main__":
    main()
