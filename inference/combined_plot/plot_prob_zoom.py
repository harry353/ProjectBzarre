from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


def plot_prob_zoom(ax: plt.Axes, prob_df: pd.DataFrame, zoom_start: pd.Timestamp, zoomed_days: int) -> None:
    prob_zoom = prob_df[prob_df["timestamp"] >= zoom_start]
    ax.plot(
        prob_zoom["timestamp"],
        prob_zoom["p_cumulative"],
        color="black",
        marker="o",
        markersize=2,
        linewidth=1.2,
        linestyle="-",
        label=None,
    )

    # Mark midnights and annotate date labels with slightly darker guide lines
    if not prob_zoom.empty:
        start_day = prob_zoom["timestamp"].min().normalize()
        end_day = prob_zoom["timestamp"].max().normalize() + pd.Timedelta(days=1)  # include last midnight
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        label_y = ymax - 0.02 * y_range  # keep labels inside plot
        for day in pd.date_range(start_day, end_day, freq="D"):
            midnight = pd.Timestamp(day)
            ax.axvline(midnight, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(
                midnight + pd.Timedelta(hours=0.5),
                label_y,
                midnight.strftime("%d-%b"),
                rotation=90,
                va="top",
                ha="left",
                fontsize=7,
                color="dimgray",
                clip_on=True,
            )

    ax.set_title(f"Storm Probability (Last {zoomed_days} Days)")
    ax.set_xlabel("Time (UTC)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
