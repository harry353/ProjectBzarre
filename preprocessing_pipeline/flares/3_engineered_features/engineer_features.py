from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from preprocessing_pipeline.utils import write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

FLARE_DB = (
    STAGE_DIR.parents[1]
    / "flares"
    / "2_hard_filtering"
    / "flares_comb_filt.db"
)
FLARE_TABLE = "filtered_flares"

OUTPUT_DB = STAGE_DIR / "flares_comb_filt_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------
START = pd.Timestamp("1998-01-01T00:00:00Z")
END = pd.Timestamp("2025-11-30T23:00:00Z")
HOURLY_INDEX = pd.date_range(START, END, freq="1h", tz="UTC")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DECAY_TAU_HOURS = 8.0
LARGE_TIME_FILL = 1e6

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_flares() -> pd.DataFrame:
    import sqlite3

    with sqlite3.connect(FLARE_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {FLARE_TABLE}",
            conn,
            parse_dates=["event_time"],
        )

    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    # ASSUMPTION: table already contains only EVENT_PEAK rows
    df = df.sort_values("event_time")
    df = df[~df["event_time"].duplicated(keep="last")]

    return df


def _flare_class_ord(peak_flux: pd.Series) -> pd.Series:
    return pd.cut(
        peak_flux,
        bins=[-np.inf, 1e-7, 1e-6, 1e-5, 1e-4, np.inf],
        labels=[0, 1, 2, 3, 4],  # A, B, C, M, X
    ).astype(int)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def engineer_flare_features() -> pd.DataFrame:
    flares = _load_flares()
    flares = flares.set_index("event_time")

    # --------------------------------------------------------------
    # Per-flare derived quantities
    # --------------------------------------------------------------
    flares["flare_class_ord"] = _flare_class_ord(flares["peak_flux_wm2"])
    flares["flare_major_flag"] = (flares["flare_class_ord"] >= 3).astype(int)
    flares["flare_extreme_flag"] = (flares["flare_class_ord"] == 4).astype(int)

    flares["energy_proxy"] = (
        flares["peak_flux_wm2"] * flares["integrated_flux"]
    ).fillna(0.0)

    prev_peak = flares["peak_flux_wm2"].shift(1)
    flares["delta_last_flare_peak_flux"] = (
        flares["peak_flux_wm2"] - prev_peak
    ).fillna(0.0)

    prev_energy = flares["energy_proxy"].shift(1)
    flares["flare_energy_ratio"] = (
        flares["energy_proxy"] / np.maximum(prev_energy, 1e-12)
    ).fillna(0.0)

    flares["flare_overtaking_flag"] = (
        flares["flare_energy_ratio"] > 5.0
    ).astype(int)

    # --------------------------------------------------------------
    # Hourly backbone
    # --------------------------------------------------------------
    hourly = pd.DataFrame(index=HOURLY_INDEX)
    hourly.index.name = "time_tag"

    # --------------------------------------------------------------
    # Event flags
    # --------------------------------------------------------------
    hourly_counts = (
        pd.Series(1, index=flares.index)
        .groupby(pd.Grouper(freq="1h"))
        .sum()
        .reindex(hourly.index, fill_value=0)
    )

    hourly["flare_active_flag"] = (hourly_counts > 0).astype(int)

    # --------------------------------------------------------------
    # Time since / until flare
    # --------------------------------------------------------------
    last_event = flares.index.to_series().reindex(hourly.index, method="ffill")
    next_event = flares.index.to_series().reindex(hourly.index, method="bfill")

    hourly["hours_since_last_flare"] = (
        (hourly.index - last_event).dt.total_seconds() / 3600
    ).fillna(LARGE_TIME_FILL)

    hourly["hours_until_next_flare"] = (
        (next_event - hourly.index).dt.total_seconds() / 3600
    ).fillna(LARGE_TIME_FILL)

    # --------------------------------------------------------------
    # Flare rate features
    # --------------------------------------------------------------
    hourly["flare_count_last_6h"] = hourly_counts.rolling(6, min_periods=1).sum()
    hourly["flare_count_last_24h"] = hourly_counts.rolling(24, min_periods=1).sum()
    hourly["flare_count_last_72h"] = hourly_counts.rolling(72, min_periods=1).sum()

    # --------------------------------------------------------------
    # Carry-forward flare properties
    # --------------------------------------------------------------
    def ff(col, fill=0.0):
        return col.reindex(hourly.index, method="ffill").fillna(fill)

    hourly["last_flare_peak_flux"] = ff(flares["peak_flux_wm2"])
    hourly["last_flare_integrated_flux"] = ff(flares["integrated_flux"])
    hourly["last_flare_background_flux"] = ff(flares["background_flux"])
    hourly["last_flare_xrsb_flux"] = ff(flares["xrsb_flux"])
    hourly["last_flare_energy_proxy"] = ff(flares["energy_proxy"])

    hourly["flare_class_ord"] = ff(flares["flare_class_ord"], 0).astype(int)
    hourly["flare_major_flag"] = ff(flares["flare_major_flag"], 0).astype(int)
    hourly["flare_extreme_flag"] = ff(flares["flare_extreme_flag"], 0).astype(int)

    hourly["delta_last_flare_peak_flux"] = ff(
        flares["delta_last_flare_peak_flux"]
    )
    hourly["flare_energy_ratio"] = ff(flares["flare_energy_ratio"])
    hourly["flare_overtaking_flag"] = ff(
        flares["flare_overtaking_flag"], 0
    ).astype(int)

    # --------------------------------------------------------------
    # Flare influence decay
    # --------------------------------------------------------------
    hourly["flare_influence_exp"] = np.exp(
        -hourly["hours_since_last_flare"] / DECAY_TAU_HOURS
    )

    # --------------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------------
    if hourly.isna().any().any():
        raise RuntimeError("NaNs detected in flare feature table.")

    write_sqlite_table(hourly, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] Flare engineered features written to {OUTPUT_DB}")
    return hourly


def main() -> None:
    engineer_flare_features()


if __name__ == "__main__":
    main()
