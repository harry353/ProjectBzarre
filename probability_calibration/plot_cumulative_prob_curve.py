from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"

SURVIVAL_DB = PROJECT_ROOT / "probability_calibration" / "survival_probabilities.db"
SURVIVAL_TABLE = "survival_probabilities"

# ---------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------
YEAR = 2024

# Choose which quantity to visualize
# Example options:
#   "p_storm_within_4h"
#   "p_storm_within_8h"
#   "survival_to_h4"
VALUE_COL = "p_storm_within_8h"

PLOT_ONLY_ABOVE = False
PLOT_THRESHOLD = 0.3

WINDOW_START = None  # pd.Timestamp("2024-05-10", tz="UTC")
WINDOW_END = None    # pd.Timestamp("2024-05-13", tz="UTC")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, dst FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = _ensure_utc_index(df.index)
    return df["dst"]


def _load_survival(year: int) -> pd.DataFrame:
    if not SURVIVAL_DB.exists():
        raise FileNotFoundError(f"Missing survival DB: {SURVIVAL_DB}")

    with sqlite3.connect(SURVIVAL_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {SURVIVAL_TABLE}", conn)

    if df.empty:
        raise RuntimeError("Survival probability table is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    df = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    if df.empty:
        raise RuntimeError(f"No survival rows found for year {year}.")

    return df.set_index("timestamp")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dst = _load_dst()
    surv = _load_survival(YEAR)

    if VALUE_COL not in surv.columns:
        raise RuntimeError(f"Missing column '{VALUE_COL}' in survival table.")

    series = surv[VALUE_COL].astype(float).clip(0.0, 1.0)

    # Align with Dst
    common_idx = dst.index.intersection(series.index)
    dst = dst.loc[common_idx]
    series = series.loc[common_idx]

    if WINDOW_START or WINDOW_END:
        dst = dst.loc[WINDOW_START:WINDOW_END]
        series = series.loc[WINDOW_START:WINDOW_END]

    if dst.empty or series.empty:
        raise RuntimeError("No overlapping Dst / survival data to plot.")

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(dst.index, dst.values, color="tab:blue", lw=1, label="Dst")

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("YlOrRd")

    for t, p in series.items():
        if PLOT_ONLY_ABOVE and p < PLOT_THRESHOLD:
            continue
        ax.axvspan(
            t,
            t + pd.Timedelta(hours=1),
            color=cmap(norm(p)),
            alpha=0.25,
            lw=0,
        )

    ax.axhline(-50, ls=":", color="black", alpha=0.4)
    ax.axhline(0, ls="-", color="black", alpha=0.3)

    ax.set_ylabel("Dst (nT)")
    ax.set_xlabel("Time")
    ax.set_title(f"Dst with {VALUE_COL.replace('_', ' ')} – {YEAR}")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02, label=VALUE_COL)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
