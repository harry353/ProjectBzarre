from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent
HOURLY_DB = STAGE_DIR.parents[1] / "sunspot_number" / "1_averaging" / "sunspot_number_aver.db"
HOURLY_TABLE = "hourly_data"
FIGURE_PATH = STAGE_DIR / "sunspot_number_missingness.png"
COLUMNS = ["sunspot_number"]


def plot_sunspot_missingness() -> None:
    df = load_hourly_output(HOURLY_DB, HOURLY_TABLE)
    if df.empty:
        raise RuntimeError("Sunspot hourly dataset is empty; run averaging first.")

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for column in COLUMNS:
        series = df.get(column)
        if series is None:
            continue
        clean = series.dropna()
        if clean.empty:
            continue
        gaps = clean.index.to_series().diff().dt.total_seconds().fillna(0.0) / 3600.0
        ax.plot(clean.index, gaps, label=column)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend()

    ax.set_title("Sunspot number time between valid samples")
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Gap (hours)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"[OK] Missingness plot written to {FIGURE_PATH}")


def main() -> None:
    plot_sunspot_missingness()


if __name__ == "__main__":
    main()
