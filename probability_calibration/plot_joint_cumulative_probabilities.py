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
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"

PROB_DB = Path(__file__).resolve().parent / "survival_probabilities.db"
PROB_TABLE = "survival_probabilities"

YEAR = 2024
PLOT_ONLY_ABOVE = False
PLOT_THRESHOLD = 0.3

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def load_dst(year: int) -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag AS t, dst FROM hourly_data",
            conn,
            parse_dates=["t"],
        )
    df = df.set_index("t").sort_index()
    df.index = _ensure_utc(df.index)

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)
    return df.loc[start:end, "dst"]


def load_probabilities(year: int) -> pd.DataFrame:
    with sqlite3.connect(PROB_DB) as conn:
        df = pd.read_sql(
            f"SELECT * FROM {PROB_TABLE}",
            conn,
            parse_dates=["timestamp"],
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    prob_cols = [
        c for c in df.columns if c.startswith("p_storm_within_")
    ]
    if not prob_cols:
        raise RuntimeError(
            f"No cumulative probability columns found. Columns: {df.columns.tolist()}"
        )

    prob_cols = sorted(
        prob_cols,
        key=lambda x: int(x.split("_")[-1].replace("h", ""))
    )

    df[prob_cols] = df[prob_cols].apply(pd.to_numeric, errors="coerce")

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    return df.loc[start:end, prob_cols]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dst = load_dst(YEAR)
    probs = load_probabilities(YEAR)

    # Align time indices
    probs = probs.reindex(dst.index)
    probs = probs.interpolate(limit=1)

    if probs.isna().all().all():
        raise RuntimeError("All probability values are NaN after alignment.")

    fig, (ax, ax_prob) = plt.subplots(
        2, 1, figsize=(15, 6), sharex=True, height_ratios=[2, 1]
    )

    # Plot Dst
    ax.plot(dst.index, dst.values, color="black", lw=1.0, label="Dst")

    # Color map for horizons
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("YlOrRd")

    horizons = [
        int(c.split("_")[-1].replace("h", ""))
        for c in probs.columns
    ]

    for col, h in zip(probs.columns, horizons):
        series = probs[col]
        if PLOT_ONLY_ABOVE:
            series = series.where(series >= PLOT_THRESHOLD)
        color = cmap(norm(float(series.max(skipna=True)) if series.notna().any() else 0.0))
        ax_prob.plot(series.index, series.values, color=color, lw=1.0, alpha=0.6)

    ax.axhline(-50, ls=":", color="gray", alpha=0.6)
    ax.axhline(0, ls=":", color="gray", alpha=0.4)

    ax.set_ylabel("Dst (nT)")
    ax_prob.set_ylabel("Cumulative probability")
    ax_prob.set_ylim(0.0, 1.0)
    ax_prob.grid(True, alpha=0.3)
    ax.set_xlabel("Time")
    ax.set_title(f"Cumulative storm probability (1–8h) over Dst – {YEAR}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
