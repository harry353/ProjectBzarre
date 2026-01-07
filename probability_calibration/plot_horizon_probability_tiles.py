from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors, patches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
CALIB_DB = PROJECT_ROOT / "probability_calibration" / "validation_calibration.db"
DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"
CALIB_TABLE = "validation"

TARGET_TIMESTAMP = "2024-05-10 16:00:00+00:00"

HORIZONS = range(1, 7)


def _load_row(ts: pd.Timestamp) -> pd.Series:
    if not CALIB_DB.exists():
        raise FileNotFoundError(f"Missing calibration DB: {CALIB_DB}")
    with sqlite3.connect(CALIB_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {CALIB_TABLE}",
            conn,
            parse_dates=["timestamp"],
        )
    if df.empty:
        raise RuntimeError("Calibration DB is empty.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    row = df.loc[df["timestamp"] == ts]
    if row.empty:
        raise RuntimeError(f"No row found for timestamp {ts}.")
    return row.iloc[0]


def _load_dst_kp(ts: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    start = ts - pd.Timedelta(hours=10)
    end = ts

    with sqlite3.connect(DST_DB) as conn:
        dst_df = pd.read_sql(
            "SELECT time_tag AS timestamp, dst FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    dst_df["timestamp"] = pd.to_datetime(dst_df["timestamp"], utc=True, errors="coerce")
    dst_df = dst_df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    dst_slice = dst_df.loc[start:end, "dst"]

    with sqlite3.connect(KP_DB) as conn:
        kp_df = pd.read_sql(
            "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    kp_df["timestamp"] = pd.to_datetime(kp_df["timestamp"], utc=True, errors="coerce")
    kp_df = kp_df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    kp_slice = kp_df.loc[start:end, "kp_index"]

    return dst_slice, kp_slice


def main() -> None:
    ts = pd.to_datetime(TARGET_TIMESTAMP, utc=True)
    row = _load_row(ts)

    probs = []
    for h in HORIZONS:
        col = f"raw_prob_h{h}"
        if col not in row:
            raise RuntimeError(f"Missing column '{col}' in calibration DB.")
        prob = float(row[col])
        probs.append(np.clip(prob, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, len(HORIZONS))
    ax.set_ylim(-0.4, 2)
    ax.axis("off")

    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap("YlOrRd")

    for idx, (h, p) in enumerate(zip(HORIZONS, probs)):
        horizon_time = ts + pd.Timedelta(hours=h)
        rect = patches.Rectangle(
            (idx, 0),
            1,
            2,
            facecolor=cmap(norm(p)),
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            idx + 0.5,
            -0.2,
            horizon_time.strftime("%H:%M"),
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )
        ax.text(
            idx + 0.5,
            0.7,
            f"{p * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    ax.set_title(f"Real-Time Storm Forecast, {ts.strftime('%d-%b-%Y')}")
    plt.tight_layout()
    plt.show()

    # dst, kp = _load_dst_kp(ts)
    # fig, (ax_dst, ax_kp) = plt.subplots(
    #     2, 1, figsize=(12, 4), sharex=True, height_ratios=[2, 1]
    # )
    # ax_dst.plot(dst.index, dst.values, color="tab:blue", lw=1)
    # ax_kp.plot(kp.index, kp.values, color="tab:gray", lw=1)

    # ax_dst.axhline(0, ls=":", color="black", alpha=0.3)
    # ax_dst.axhline(-50, ls=":", color="black", alpha=0.3)
    # ax_kp.axhline(5, ls=":", color="black", alpha=0.3)

    # ax_dst.set_ylabel("Dst (nT)")
    # ax_kp.set_ylabel("Kp")
    # ax_kp.set_xlabel("Time")
    # ax_dst.set_title(
    #     f"Dst and Kp from {(ts - pd.Timedelta(hours=10)).strftime('%H:%M')} "
    #     f"to {ts.strftime('%H:%M')} UTC"
    # )

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
