from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from inference.combined_plot.constants import STORM_COLOR, CALM_COLOR, DST_THRESHOLD


def plot_dst_main(
    ax: plt.Axes,
    dst_hist: pd.DataFrame,
    ts_col_dst: str,
    dst_col: str,
    hist_segments: list[dict],
    forecast_times: list[pd.Timestamp],
    q10_list: list[float],
    q50_list: list[float],
    q90_list: list[float],
    anchor_ts: pd.Timestamp,
    anchor_dst: float,
    regime: str,
) -> None:
    color = STORM_COLOR if regime == "storm" else CALM_COLOR

    ax.plot(
        dst_hist[ts_col_dst],
        dst_hist[dst_col],
        label="DST (actual)",
        color="black",
        linewidth=1.2,
    )
    ax.axhline(0, color="dimgray", linewidth=1.0, linestyle="-", alpha=0.8)
    ax.axhline(DST_THRESHOLD, color="dimgray", linewidth=1.0, linestyle="--", alpha=0.8)

    if forecast_times:
        ax.fill_between(
            forecast_times,
            q10_list,
            q90_list,
            color=color,
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=None,
        )
        ax.plot(
            forecast_times,
            q50_list,
            color=color,
            linestyle="--",
            label=f"DST forecast $q_{{0.5}}$ ({regime})",
        )
        ax.plot(
            [anchor_ts, forecast_times[0]],
            [anchor_dst, q50_list[0]],
            color="black",
            linewidth=1.2,
            linestyle="-",
            label=None,
        )

    label_used = {}
    for seg in sorted(hist_segments, key=lambda s: pd.to_datetime(s["start"])):
        key = seg["regime"]
        lbl = None if label_used.get(key) else f"Issued $q_{{0.1}}$–$q_{{0.9}}$ ({key})"
        ax.fill_between(
            seg["ts"],
            seg["q10"],
            seg["q90"],
            color=seg["color"],
            alpha=0.3,
            linewidth=0,
            edgecolor="none",
            step=None,
            label=lbl,
        )
        label_used[key] = True

    ax.set_title("DST Forecast (Last 14 Days)")
    ax.set_ylabel("Dst (nT)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
