from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_cme(df: pd.DataFrame, title_suffix: str | None = None):
    """
    Scatter DONKI CME speeds using the same style as the LASCO plot so that
    downstream comparisons share the exact layout.
    """
    if df.empty:
        print("No DONKI CME data available to plot.")
        return

    frame = df.dropna(subset=["time21_5"]).sort_values("time21_5")

    if frame.empty:
        return

    plt.figure(figsize=(12, 4))
    plt.scatter(frame["time21_5"], frame.get("speed"), marker="o", linewidths=1.5)
    title = "DONKI CME Speeds"
    if title_suffix:
        title = f"{title} ({title_suffix})"

    plt.title(title)
    plt.xlabel("Event Time (UTC)")
    plt.ylabel("Speed (km/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
