from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_prob_main(ax: plt.Axes, prob_df: pd.DataFrame, history_hours: int) -> None:
    ax.plot(
        prob_df["timestamp"],
        prob_df["p_cumulative"],
        color="black",
        marker="o",
        markersize=2,
        linewidth=1.2,
        linestyle="-",
        label="Storm probability",
    )
    ax.set_title(f"Storm Probability (Last {history_hours // 24} Days)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Probability")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%Y"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
