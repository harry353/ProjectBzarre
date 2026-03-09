from __future__ import annotations
import sys
from pathlib import Path

# Handle absolute path resolution with fallbacks for local and installed project layouts
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Ensure the project root is in the system path for local module imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

# Force non-interactive Agg backend to avoid GUI dependencies during pipeline execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import numpy as np

from preprocessing_pipeline.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent
# Source the combined multi-instrument database from the previous stage
HOURLY_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "2_concatenating_combining"
    / "imf_solar_wind_aver_comb.db"
)
HOURLY_TABLE = "hourly_data"
FIGURE_PATH = STAGE_DIR / "imf_solar_wind_missingness.png"
# Logical grouping of parameters for visualization clarity
GROUPS = {
    "Solar wind": ["density", "speed", "temperature"],
    "IMF": ["bx_gse", "by_gse", "bz_gse", "bt"],
}


# Visualizes the temporal gaps in the IMF and Solar Wind telemetry
def plot_imf_sw_missingness() -> None:
    df = load_hourly_output(HOURLY_DB, HOURLY_TABLE)
    if df.empty:
        raise RuntimeError("IMF + solar wind dataset is empty; run earlier stages first.")

    # Create vertically stacked subplots for each parameter group
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(12, 6), sharex=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    for ax, (title, columns) in zip(axes, GROUPS.items()):
        plotted = False
        for column in columns:
            series = df.get(column)
            if series is None:
                continue
            # Isolate valid (non-NaN) samples to identify observation gaps
            clean = series.dropna()
            if clean.empty:
                continue
            # Calculate time difference between consecutive valid samples in decimal hours
            gaps = clean.index.to_series().diff().dt.total_seconds().fillna(0.0) / 3600.0
            ax.plot(clean.index, gaps, label=column)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.legend()
        ax.set_title(title)
        ax.set_ylabel("Gap (hours)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestamp (UTC)")
    fig.suptitle("IMF + solar wind time between valid samples")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    # Persist the diagnostic plot as a PNG
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"[OK] Missingness plot written to {FIGURE_PATH}")


def main() -> None:
    plot_imf_sw_missingness()


if __name__ == "__main__":
    main()
