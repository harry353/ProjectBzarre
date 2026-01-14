from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from scipy.signal import find_peaks


STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR
for parent in STAGE_DIR.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = STAGE_DIR.parent


PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"
DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"
HAZARD_DB = STAGE_DIR / "storm_onset_hazards.db"
HAZARD_TABLES = [
    "storm_onset_train",
    "storm_onset_validation",
    "storm_onset_test",
]

YEAR_TO_PLOT = 2024
HORIZON_TO_PLOT = 8
OUTPUT_PATH: Path | None = None
PEAK_PROMINENCE = 39.0


def _ensure_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize("UTC")
        if series.dt.tz is None
        else series.dt.tz_convert("UTC")
    )


def _load_dst() -> pd.DataFrame:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, dst FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.rename(columns={"dst": "dst_phys"}).set_index("timestamp").sort_index()


def _load_kp() -> pd.DataFrame:
    with sqlite3.connect(KP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.rename(columns={"kp_index": "kp"}).set_index("timestamp").sort_index()


def _find_kp_peaks(series: pd.Series) -> list[pd.Timestamp]:
    values = series.to_numpy()
    peaks, _ = find_peaks(values, prominence=PEAK_PROMINENCE)
    return [series.index[i] for i in peaks]


def _load_hazard_labels() -> pd.DataFrame:
    frames = []
    with sqlite3.connect(HAZARD_DB) as conn:
        for table in HAZARD_TABLES:
            exists = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            )
            if not exists.empty:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table}", conn, parse_dates=["timestamp"]
                )
                frames.append(df)
    if not frames:
        raise RuntimeError("Hazard labels not found.")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return combined


def plot_kp_year(year: int, horizon: int, output: Path | None = None) -> None:
    dst = _load_dst()
    kp = _load_kp()
    hazards = _load_hazard_labels()

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[start:end]
    kp_slice = kp.loc[start:end]
    hazards_slice = hazards.loc[
        (hazards["timestamp"] >= start) & (hazards["timestamp"] < end)
    ]

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], label="Dst")
    ax_kp.plot(kp_slice.index, kp_slice["kp"], label="Kp")

    col = f"h_{horizon}"
    mask = hazards_slice[col] == 1
    spans = hazards_slice.loc[mask, "timestamp"]

    if not spans.empty:
        step = pd.Timedelta(hours=1)
        groups = (spans.diff() > step).cumsum()
        for _, g in spans.groupby(groups):
            ax_dst.axvspan(g.iloc[0], g.iloc[-1] + step, color="red", alpha=0.25)
            ax_kp.axvspan(g.iloc[0], g.iloc[-1] + step, color="red", alpha=0.2)

    for peak in _find_kp_peaks(kp_slice["kp"]):
        ax_dst.axvline(peak, color="gray", linestyle="--", alpha=0.6)
        ax_kp.axvline(peak, color="gray", linestyle="--", alpha=0.6)

    ax_dst.axhline(0, color="black", alpha=0.5)
    ax_dst.axhline(-50, color="black", linestyle=":", alpha=0.3)
    ax_kp.axhline(5, color="black", linestyle=":", alpha=0.3)

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")
    ax_dst.set_title(f"DST and Kp with interval hazard labels (h{horizon}) in {year}")

    ax_dst.grid(alpha=0.3)
    ax_kp.grid(alpha=0.3)

    ax_dst.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", label="Dst"),
            Line2D([0], [0], color="red", linewidth=2, label="Storm interval"),
            Line2D([0], [0], color="gray", linestyle="--", label="Kp peak"),
            Line2D([0], [0], color="black", linestyle=":", label="-50 nT"),
        ]
    )

    ax_kp.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", label="Kp"),
            Line2D([0], [0], color="black", linestyle=":", label="Kp = 5"),
        ]
    )

    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    plot_kp_year(YEAR_TO_PLOT, HORIZON_TO_PLOT, OUTPUT_PATH)


if __name__ == "__main__":
    main()
