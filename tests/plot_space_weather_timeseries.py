from __future__ import annotations

import sys
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "preprocessing_pipeline" / "space_weather.db"
YEAR_TO_PLOT = 2024


def _load_timeseries(table: str, value_col: str, *, method: str = "mean") -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"SELECT time_tag, {value_col} FROM {table}",
            conn,
            parse_dates=["time_tag"],
        )
    if df.empty:
        return df
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df = df.dropna(subset=["time_tag"])
    df = df.set_index("time_tag").sort_index()
    df = df[df.index.year == YEAR_TO_PLOT]
    if method == "ffill":
        hourly = df.resample("1h").ffill()
    else:
        hourly = df.resample("1h").mean()
    return hourly


def plot_space_weather() -> None:
    dst = _load_timeseries("dst_index", "dst")
    kp = None
    sw_density = _load_timeseries("dscovr_f1m", "density", method="ffill")
    sw_speed = _load_timeseries("dscovr_f1m", "speed", method="ffill")
    sw_temp = _load_timeseries("dscovr_f1m", "temperature", method="ffill")
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(dst.index, dst["dst"], color="black", linewidth=0.8)
    axes[0].set_title("Dst")
    axes[0].set_ylabel("nT")

    axes[1].plot(sw_density.index, sw_density["density"], color="#1f77b4", linewidth=0.8)
    axes[1].set_title("DSCOVR Solar Wind Density")
    axes[1].set_ylabel("cm⁻³")

    axes[2].plot(sw_speed.index, sw_speed["speed"], color="#ff7f0e", linewidth=0.8)
    axes[2].set_title("DSCOVR Solar Wind Speed")
    axes[2].set_ylabel("km/s")

    axes[3].plot(sw_temp.index, sw_temp["temperature"], color="#2ca02c", linewidth=0.8)
    axes[3].set_title("DSCOVR Solar Wind Temperature")
    axes[3].set_ylabel("K")
    for ax in axes:
        ax.grid(True, alpha=0.3, linewidth=0.6)

    axes[3].set_xlabel("Time (UTC)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing database: {DB_PATH}")
    plot_space_weather()


if __name__ == "__main__":
    main()
