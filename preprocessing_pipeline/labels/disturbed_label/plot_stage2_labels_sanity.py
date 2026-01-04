from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve()
for p in PROJECT_ROOT.parents:
    if (p / "space_weather_api.py").exists():
        PROJECT_ROOT = p
        break

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"

STAGE2_DB = Path(__file__).resolve().parent / "stage2_labels.db"
STAGE2_TABLE = "stage2_labels"

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

YEAR = 2024
LEAD_HOURS = 8   # MUST match label construction

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def ensure_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql(
            "SELECT time_tag, dst FROM hourly_data",
            conn,
            parse_dates=["time_tag"],
        )
    df = df.set_index("time_tag").sort_index()
    df.index = ensure_utc(df.index)
    return df["dst"]


def load_stage2_labels() -> pd.Series:
    with sqlite3.connect(STAGE2_DB) as conn:
        df = pd.read_sql(
            f"""
            SELECT time_tag, stage2_storm_label
            FROM {STAGE2_TABLE}
            """,
            conn,
            parse_dates=["time_tag"],
        )
    df = df.set_index("time_tag").sort_index()
    df.index = ensure_utc(df.index)
    return df["stage2_storm_label"]

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    dst = load_dst()
    labels = load_stage2_labels()

    start = pd.Timestamp(f"{YEAR}-01-01", tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst = dst.loc[start:end]
    labels = labels.loc[start:end]

    # Align just in case
    idx = dst.index.intersection(labels.index)
    dst = dst.loc[idx]
    labels = labels.loc[idx]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ---- Dst ----
    ax1.plot(dst.index, dst, color="black", lw=1)
    ax1.axhline(-50, ls=":", color="red", alpha=0.6)
    ax1.axhline(-100, ls=":", color="gray", alpha=0.4)

    # ---- Stage-2 shading ----
    for t, flag in labels.items():
        if flag == 1:
            ax1.axvspan(
                t,
                t + pd.Timedelta(hours=LEAD_HOURS),
                color="#e74c3c",
                alpha=0.20,
                lw=0,
            )

    # ---- Stage-2 label series ----
    ax2.step(
        labels.index,
        labels,
        where="post",
        color="black",
        lw=1.5,
    )
    ax2.set_yticks([0, 1])
    ax2.set_ylabel("Stage-2\nLabel")

    # ---- Labels ----
    ax1.set_ylabel("Dst (nT)")
    ax1.set_title(
        f"Stage-2 storm labels sanity check ({YEAR})\n"
        f"Red shading = storm occurs within next {LEAD_HOURS} hours"
    )
    ax2.set_xlabel("Time")

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
