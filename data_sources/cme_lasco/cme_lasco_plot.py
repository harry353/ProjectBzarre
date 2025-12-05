from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_cme_lasco(df: pd.DataFrame, title_suffix: str | None = None) -> None:
    """
    Scatter LASCO CME linear speeds for the provided dataframe.
    """
    if df.empty:
        print("No LASCO CME data available to plot.")
        return

    frame = df.dropna(subset=["Datetime"]).sort_values("Datetime")

    if frame.empty:
        print("No LASCO CME entries include valid timestamps to plot.")
        return

    plt.figure(figsize=(12, 4))
    plt.scatter(frame["Datetime"], frame["Linear_Speed"], marker="o", linewidths=1.5)
    title = "LASCO CME Linear Speeds"
    if title_suffix:
        title = f"{title} ({title_suffix})"

    plt.title(title)
    plt.xlabel("Event Time (UTC)")
    plt.ylabel("Linear Speed (km/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
