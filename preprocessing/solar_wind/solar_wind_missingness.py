from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.hourly_utils import load_hourly_output

OUTPUT_DB = Path(__file__).resolve().parent / "solar_wind_hourly.db"
TABLES = [
    ("ace_swepam_hourly", "ACE SWEPAM"),
    ("dscovr_f1m_hourly", "DSCOVR F1M"),
    ("solar_wind_hourly", "Combined Solar Wind"),
]
COLUMNS = ["density", "speed", "temperature"]


def main() -> None:
    datasets: list[pd.DataFrame] = []
    labels: list[str] = []
    for table_name, label in TABLES:
        try:
            df = load_hourly_output(OUTPUT_DB, table_name)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARN] {exc}")
            continue
        if df.empty:
            print(f"[WARN] Hourly table '{table_name}' empty for missingness plot.")
            continue
        datasets.append(df)
        labels.append(label)

    if not datasets:
        print("[WARN] No solar wind missingness plots were generated.")
        return

    fig, axes = plt.subplots(len(COLUMNS), 1, figsize=(12, 8), sharex=True)
    for ax, column in zip(axes, COLUMNS):
        _scatter_gap_plot(ax, datasets, labels, column)

    axes[-1].set_xlabel("Timestamp (UTC)")
    plt.tight_layout()
    plt.show()


def _scatter_gap_plot(ax, datasets: list[pd.DataFrame], labels: list[str], column: str) -> None:
    plotted = False
    for df, label in zip(datasets, labels):
        series = df.get(column)
        if series is None:
            continue
        clean = series.dropna()
        if clean.empty:
            continue
        times = clean.index
        gaps = pd.Series(times).diff().dt.total_seconds().fillna(0) / 3600.0
        ax.scatter(times, gaps, s=10, label=label)
        plotted = True

    ax.set_title(f"{column.title()} hourly gaps")
    ax.set_ylabel("Gap (hours)")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid samples", transform=ax.transAxes, ha="center")


if __name__ == "__main__":
    main()
