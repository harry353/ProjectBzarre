"""Plotting helpers for GOES flare archive data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_flares_archive(df, title_suffix=""):
    if df.empty:
        print("[WARN] Flare archive dataframe empty. Cannot plot.")
        return

    working = df.copy().sort_values("event_time")
    times = pd.to_datetime(working["event_time"])
    flux = working["peak_flux_wm2"].astype(float)
    classes = working["flare_class"].fillna("")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(times, flux, c="#0288d1", s=35, edgecolor="black", linewidth=0.4)

    ax.set_yscale("log")
    ax.set_ylabel("Peak Flux (W/m^2)")
    ax.set_xlabel("Time (UTC)")
    title = "GOES Flare Archive"
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
