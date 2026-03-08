from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "classification_pipeline_" / "horizon_models"

CALIB_TABLE = "calibrated_probs"
TARGET_HORIZONS_H = range(1, 9)

DATETIME_STR = "2024-05-10 17:00:00"
SPLIT = "test"


# Retrieves calibrated interval probabilities for all target horizons @ a specific UTC timestamp
def _load_probs_at_timestamp(ts: pd.Timestamp) -> list[float]:
    probs: list[float] = []

    for h in TARGET_HORIZONS_H:
        db = MODEL_ROOT / f"h{h}" / "calibrated_probabilities.db"
        if not db.exists():
            raise RuntimeError(f"Missing calibrated DB for h{h}")

        with sqlite3.connect(db) as conn:
            # Query probabilities for the specific split and timestamp
            df = pd.read_sql_query(
                f"""
                SELECT prob_calibrated
                FROM {CALIB_TABLE}
                WHERE timestamp = ?
                  AND split = ?
                """,
                conn,
                params=(
                    ts.strftime("%Y-%m-%d %H:%M:%S"),
                    SPLIT,
                ),
            )

        if df.empty:
            raise RuntimeError(f"No calibrated probability for h{h} at {ts}")

        probs.append(float(df.iloc[0]["prob_calibrated"]))

    return probs


# Main entry point to load probabilities and render the tile-based visualization
def main() -> None:
    ts = pd.to_datetime(DATETIME_STR, utc=True)
    probs = _load_probs_at_timestamp(ts)

    # 1. Visualization configuration
    # Use Yellow-Orange-Brown colormap for probability intensity (0 to 1 scale)
    cmap = plt.cm.YlOrBr
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    # 2. Layout Initialization
    # Create a horizontal row of tiles, one for each forecast horizon
    fig, axes = plt.subplots(1, len(TARGET_HORIZONS_H), figsize=(14, 2))

    # 3. Tile Rendering Loop
    for idx, ax in enumerate(axes):
        value = probs[idx]
        color = cmap(norm(value))

        # Render each tile with background color corresponding to probability
        ax.set_facecolor(color)
        ax.text(
            0.5,
            0.5,
            f"h{idx + 1}\n{value:.3f}",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )
        # Remove axes decorations for a cleaner "tile" dashboard appearance
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"Calibrated storm-onset interval probabilities @ {ts}",
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
