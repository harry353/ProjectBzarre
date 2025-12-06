"""Diagnostic plotting for GOES flare summary data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_flares(df, title_suffix=""):
    """Scatter peak flux values coloured by flare class."""

    if df.empty:
        print("[WARN] Flare dataframe empty. Cannot plot.")
        return

    working = df.copy()
    working = working.sort_values("event_time")

    times = pd.to_datetime(working["event_time"])
    flux = working["peak_flux_wm2"].astype(float)
    classes = working["flare_class"].fillna("")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(times, flux, c="#ff6f00", s=35, edgecolor="k", linewidth=0.4)

    ax.set_yscale("log")
    ax.set_ylabel("Peak Flux (W/m^2)")
    ax.set_xlabel("Time (UTC)")
    title = "GOES Flare Classes"
    if title_suffix:
        title = f"{title} — {title_suffix}"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    for x, y, label in zip(times, flux, classes):
        if np.isnan(y):
            continue
        ax.text(x, y * 1.05, label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()
