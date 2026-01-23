from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
    ax.set_title(f"Storm Probability (Last {zoomed_days} Days)")
    ax.set_xlabel("Timestamp")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
