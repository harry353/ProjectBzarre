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
YEAR_TO_PLOT = 2024
HORIZON_TO_PLOT = 8
OUTPUT_PATH: Path | None = None
PEAK_PROMINENCE = 39.0
KP_TO_AP = {
    0.00: 0, 0.33: 2, 0.67: 3, 1.00: 4, 1.33: 5, 1.67: 6, 2.00: 7,
    2.33: 9, 2.67: 12, 3.00: 15, 3.33: 18, 3.67: 22, 4.00: 27,
    4.33: 32, 4.67: 39, 5.00: 48, 5.33: 56, 5.67: 67, 6.00: 80,
    6.33: 94, 6.67: 111, 7.00: 132, 7.33: 154, 7.67: 179, 8.00: 207,
    8.33: 236, 8.67: 300, 9.00: 400,
}
KP_KEYS = pd.Series(sorted(KP_TO_AP.keys()), dtype=float)


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


def _kp_to_ap(kp: pd.Series) -> pd.Series:
    kp_vals = kp.astype(float).round(2)
    mapped = kp_vals.map(KP_TO_AP)
    missing = mapped.isna()
    if missing.any():
        vals = kp_vals[missing].to_numpy()
        idx = (vals[:, None] - KP_KEYS.to_numpy()[None, :]).__abs__().argmin(axis=1)
        mapped.loc[missing] = [KP_TO_AP[KP_KEYS.iloc[i]] for i in idx]
    return mapped.astype(float)


def _find_kp_peaks(series: pd.Series) -> list[pd.Timestamp]:
    values = series.to_numpy()
    peaks, _ = find_peaks(values, prominence=PEAK_PROMINENCE)
    return [series.index[i] for i in peaks]


def _storm_intervals(dst_series: pd.Series, ap_series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    idx = dst_series.index.intersection(ap_series.index)
    dst_series = dst_series.loc[idx]
    ap_series = ap_series.loc[idx]

    storm_mask = (ap_series > 39) & (dst_series < -50)
    if not storm_mask.any():
        return []
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    groups = (storm_mask != storm_mask.shift()).cumsum()
    prev = dst_series.shift(1)
    crossings = dst_series.index[(dst_series <= 0) & (prev > 0)]
    for _, seg in dst_series[storm_mask].groupby(groups[storm_mask]):
        start = seg.index[0]
        prior = crossings[crossings <= start]
        if len(prior) > 0:
            start = prior[-1]
        min_time = seg.idxmin()
        intervals.append((start, min_time))
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [intervals[0]]
    step = pd.Timedelta(hours=1)
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + step:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def plot_kp_year(year: int, horizon: int, output: Path | None = None) -> None:
    dst = _load_dst()
    kp = _load_kp()

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[start:end]
    kp_slice = kp.loc[start:end]
    ap_slice = _kp_to_ap(kp_slice["kp"])

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], label="Dst")
    ax_kp.plot(kp_slice.index, kp_slice["kp"], label="Kp")

    intervals = _storm_intervals(dst_slice["dst_phys"], ap_slice)
    for start_time, end_time in intervals:
        if end_time < start_time:
            continue
        ax_dst.axvspan(start_time, end_time, color="red", alpha=0.25)
        ax_kp.axvspan(start_time, end_time, color="red", alpha=0.2)

    for peak in _find_kp_peaks(kp_slice["kp"]):
        ax_dst.axvline(peak, color="gray", linestyle="--", alpha=0.6)
        ax_kp.axvline(peak, color="gray", linestyle="--", alpha=0.6)

    ax_dst.axhline(0, color="black", alpha=0.5)
    ax_dst.axhline(-50, color="black", linestyle=":", alpha=0.3)
    ax_kp.axhline(5, color="black", linestyle=":", alpha=0.3)

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")
    ax_dst.set_title(f"DST and Kp storm intervals in {year} (ap>39, dst<-50)")

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
