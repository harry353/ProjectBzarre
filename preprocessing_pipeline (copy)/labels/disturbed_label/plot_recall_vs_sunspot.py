from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR
for parent in STAGE_DIR.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = STAGE_DIR.parent

SUNSPOT_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "sunspot_number"
    / "1_averaging"
    / "sunspot_number_aver.db"
)
SUNSPOT_TABLE = "hourly_data"

OUTPUT_PATH: Path | None = None

RECALL_12H = {
    1999: 0.941,
    2000: 0.760,
    2001: 0.950,
    2002: 0.938,
    2003: 0.917,
    2004: 1.000,
    2005: 1.000,
    2006: 0.750,
    2007: 0.750,
    2008: 1.000,
    2009: 1.000,
    2010: 0.750,
    2011: 0.900,
    2012: 0.824,
    2013: 0.786,
    2014: 0.800,
    2015: 0.905,
    2016: 0.857,
    2017: 0.929,
    2018: 0.667,
    2019: 0.800,
    2020: 0.000,
    2021: 0.750,
    2022: 0.824,
    2023: 0.842,
    2024: 0.941,
    2025: 0.864,
}


def _ensure_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize("UTC")
        if series.dt.tz is None
        else series.dt.tz_convert("UTC")
    )


def _load_sunspot_hourly() -> pd.DataFrame:
    with sqlite3.connect(SUNSPOT_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT time_tag AS timestamp, sunspot_number FROM {SUNSPOT_TABLE}",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def _yearly_median_sunspot(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["year"] = df.index.year
    return df.groupby("year")["sunspot_number"].median()


def plot_recall_vs_sunspot(output: Path | None = None) -> None:
    sunspot = _load_sunspot_hourly()
    sunspot_median = _yearly_median_sunspot(sunspot)

    recall_series = pd.Series(RECALL_12H, name="recall_12h").sort_index()
    combined = pd.DataFrame({"recall_12h": recall_series})
    combined["sunspot_median"] = sunspot_median.reindex(combined.index)

    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(
        combined.index,
        combined["recall_12h"],
        color="#1f77b4",
        marker="o",
        label="Recall (12h)",
    )
    ax_right.plot(
        combined.index,
        combined["sunspot_median"],
        color="#ff7f0e",
        marker="s",
        label="Sunspot median",
    )

    ax_left.set_ylabel("Recall (12h)")
    ax_right.set_ylabel("Median sunspot number")
    ax_left.set_xlabel("Year")
    ax_left.set_title("Recall vs yearly median sunspot number")
    ax_left.grid(True, alpha=0.3)

    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, frameon=False)

    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        plt.close(fig)
        print(f"[OK] Figure saved to {output}")
    else:
        plt.show()
        plt.close(fig)


def main() -> None:
    plot_recall_vs_sunspot(OUTPUT_PATH)


if __name__ == "__main__":
    main()
