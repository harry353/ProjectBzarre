from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

STAGE_DIR = Path(__file__).resolve().parent
DST_DB = STAGE_DIR.parents[1] / "dst" / "1_averaging" / "dst_aver.db"
AP_DB = (
    STAGE_DIR.parents[1]
    / "kp_index"
    / "5_engineered_features"
    / "kp_index_aver_filt_imp_eng.db"
)

YEAR_TO_PLOT = 2005
OUTPUT_PATH: Path | None = None
PEAK_PROMINENCE = 39.0
PEAK_WINDOW_HOURS = 6
STORM_OVERLAP_ALLOWANCE = pd.Timedelta(hours=6)


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


def _load_ap() -> pd.DataFrame:
    with sqlite3.connect(AP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, ap FROM engineered_features",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.rename(columns={"ap": "ap"}).set_index("timestamp").sort_index()


def _find_peak_times(series: pd.Series) -> list[pd.Timestamp]:
    values = series.to_numpy()
    peaks, _ = find_peaks(values, prominence=PEAK_PROMINENCE)
    return [series.index[idx] for idx in peaks]


def _compute_dst_ranges(
    dst_series: pd.Series, peak_times: list[pd.Timestamp]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for peak_time in peak_times:
        past = dst_series[dst_series.index < peak_time]
        if past.empty:
            continue
        last_positive = past[past > 0].dropna()
        start = last_positive.index[-1] if not last_positive.empty else peak_time

        future = dst_series[dst_series.index >= peak_time]
        if future.empty:
            continue
        next_positive = future[future > 0]
        if next_positive.empty:
            continue
        end = next_positive.index[0]
        if end <= start:
            continue
        ranges.append((start, end))
    return ranges


def plot_ap_year(year: int, output: Path | None = None) -> None:
    dst = _load_dst()
    ap = _load_ap()

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[(dst.index >= start) & (dst.index < end)]
    ap_slice = ap.loc[(ap.index >= start) & (ap.index < end)]

    if dst_slice.empty or ap_slice.empty:
        raise ValueError(f"No overlapping DST/Ap data available for year {year}.")

    peak_times = _find_peak_times(ap_slice["ap"])
    raw_ranges = _compute_dst_ranges(dst_slice["dst_phys"], peak_times)
    dst_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in raw_ranges:
        overlap = False
        for existing_start, existing_end in dst_ranges:
            overlap_duration = min(end, existing_end) - max(start, existing_start)
            if overlap_duration >= STORM_OVERLAP_ALLOWANCE:
                overlap = True
                break
        if not overlap:
            dst_ranges.append((start, end))

    fig, (ax_dst, ax_ap) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1]
    )
    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], color="tab:blue", label="Dst")
    ax_ap.plot(ap_slice.index, ap_slice["ap"], color="tab:blue", label="Ap index")

    shaded_label_used = False
    for start, end in dst_ranges:
        label = "Ap peak range" if not shaded_label_used else None
        ax_dst.axvspan(start, end, color="lightblue", alpha=0.3, label=label)
        shaded_label_used = True

    line_label_used = False
    for peak_time in peak_times:
        label = "Ap peak" if not line_label_used else None
        ax_dst.axvline(peak_time, color="#777777", linestyle="--", label=label)
        ax_ap.axvline(peak_time, color="#777777", linestyle="--", label=None)
        line_label_used = True

    ax_dst.set_ylabel("Dst (nT)")
    ax_ap.set_ylabel("Ap")
    ax_ap.set_xlabel("Time")
    ax_dst.set_title(f"DST and Ap index in {year}")
    ax_dst.grid(True, alpha=0.3)
    ax_ap.grid(True, alpha=0.3)
    ax_dst.legend()
    ax_ap.legend()

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
    plot_ap_year(YEAR_TO_PLOT, OUTPUT_PATH)


if __name__ == "__main__":
    main()
