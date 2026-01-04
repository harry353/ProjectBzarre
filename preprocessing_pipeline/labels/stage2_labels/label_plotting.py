from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR
for parent in STAGE_DIR.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = STAGE_DIR.parent

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"

DISTURBED_DB = PIPELINE_ROOT / "features_targets" / "disturbed_label" / "disturbed_labels.db"
STAGE2_DB = PIPELINE_ROOT / "features_targets" / "stage2_labels" / "stage2_labels.db"

SSC_DB = PIPELINE_ROOT / "features_targets" / "full_storm_label" / "full_storm_labels.db"

YEAR_TO_PLOT = 2024
STAGE2_LABEL = "storm_within_8h"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _ensure_utc(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")


def _load_table(db: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def load_dst() -> pd.DataFrame:
    return _load_table(
        DST_DB,
        "SELECT time_tag AS timestamp, dst FROM hourly_data",
    )


def load_kp() -> pd.DataFrame:
    return _load_table(
        KP_DB,
        "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
    )


def load_disturbed() -> pd.Series:
    frames = []
    with sqlite3.connect(DISTURBED_DB) as conn:
        for tbl in ("disturbed_train", "disturbed_validation", "disturbed_test"):
            frames.append(
                pd.read_sql_query(
                    f"SELECT time_tag AS timestamp, disturbed_flag FROM {tbl}",
                    conn,
                    parse_dates=["timestamp"],
                )
            )
    df = pd.concat(frames).drop_duplicates("timestamp")
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp").sort_index()["disturbed_flag"]


def load_stage2_labels() -> pd.Series:
    frames = []
    with sqlite3.connect(STAGE2_DB) as conn:
        for tbl in ("stage2_train", "stage2_validation", "stage2_test"):
            frames.append(
                pd.read_sql_query(
                    f"SELECT timestamp, {STAGE2_LABEL} FROM {tbl}",
                    conn,
                    parse_dates=["timestamp"],
                )
            )
    df = pd.concat(frames).drop_duplicates("timestamp")
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp").sort_index()[STAGE2_LABEL]


def load_storm_onsets() -> list[pd.Timestamp]:
    frames = []
    with sqlite3.connect(SSC_DB) as conn:
        for tbl in (
            "storm_full_storm_train",
            "storm_full_storm_validation",
            "storm_full_storm_test",
        ):
            frames.append(
                pd.read_sql_query(
                    f"SELECT timestamp, storm_flag FROM {tbl}",
                    conn,
                    parse_dates=["timestamp"],
                )
            )
    df = pd.concat(frames).drop_duplicates("timestamp")
    df["timestamp"] = _ensure_utc(df["timestamp"])
    df = df.set_index("timestamp")

    active = df["storm_flag"] == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()

    onsets = []
    for run, flag in active.groupby(run_id):
        if flag.iloc[0]:
            onsets.append(df.index[run_id == run][0])
    return onsets


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def main() -> None:
    dst = load_dst()
    kp = load_kp()
    disturbed = load_disturbed()
    stage2 = load_stage2_labels()
    storm_onsets = load_storm_onsets()

    start = pd.Timestamp(year=YEAR_TO_PLOT, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst = dst.loc[start:end]
    kp = kp.loc[start:end]
    disturbed = disturbed.loc[start:end]
    stage2 = stage2.loc[start:end]
    storm_onsets = [t for t in storm_onsets if start <= t < end]

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax_dst.plot(dst.index, dst["dst"], color="tab:blue")
    ax_kp.plot(kp.index, kp["kp_index"], color="tab:gray")

    # Stage-1 disturbed shading
    active = disturbed == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()
    for run, flag in active.groupby(run_id):
        if flag.iloc[0]:
            idx = disturbed.index[run_id == run]
            ax_dst.axvspan(idx[0], idx[-1], color="#f1c40f", alpha=0.15)
            ax_kp.axvspan(idx[0], idx[-1], color="#f1c40f", alpha=0.15)

    # Stage-2 positives
    for t in stage2[stage2 == 1].index:
        ax_dst.axvline(t, color="red", alpha=0.25, linewidth=1)
        ax_kp.axvline(t, color="red", alpha=0.25, linewidth=1)

    # Storm onsets
    for t in storm_onsets:
        ax_dst.axvline(t, color="black", linewidth=1.5)
        ax_kp.axvline(t, color="black", linewidth=1.5)

    ax_dst.axhline(-50, linestyle=":", color="black", alpha=0.4)
    ax_kp.axhline(5, linestyle=":", color="black", alpha=0.4)

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")

    ax_dst.set_title(f"Stage-2 sanity check ({YEAR_TO_PLOT})")

    legend = [
        Patch(facecolor="#f1c40f", alpha=0.15, label="Stage-1 disturbed"),
        Patch(facecolor="red", alpha=0.25, label="Stage-2 positive (≤8h)"),
        Patch(facecolor="black", alpha=1.0, label="Storm onset"),
    ]
    ax_dst.legend(handles=legend, frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

