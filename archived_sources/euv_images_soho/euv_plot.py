from __future__ import annotations

import matplotlib.pyplot as plt


def plot_euv_metadata(metadata: dict) -> None:
    """
    Plot a simple bar chart highlighting the stored metadata columns.
    """
    labels = ["rsun_arcsec", "crpix1", "crpix2", "cdelt1", "cdelt2"]
    values = [metadata[label] for label in labels]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color="tab:blue")
    plt.title("SOHO/EIT Metadata Summary")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()
