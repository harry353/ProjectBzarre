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
SSC_DB = STAGE_DIR / "full_storm_labels.db"
SSC_TABLES = [
    "storm_full_storm_train",
    "storm_full_storm_validation",
    "storm_full_storm_test",
]

YEAR_TO_PLOT = 2024
OUTPUT_PATH: Path | None = None


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
    peaks, _ = find_peaks(values, prominence=39.0)
    return [series.index[idx] for idx in peaks]


def _load_full_storm_labels() -> pd.DataFrame:
    frames = []
    with sqlite3.connect(SSC_DB) as conn:
        for table in SSC_TABLES:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table}", conn, parse_dates=["timestamp"]
                )
                frames.append(df)
    if not frames:
        raise RuntimeError("Full-storm labels not found.")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    if combined["timestamp"].dt.tz is None:
        combined["timestamp"] = combined["timestamp"].dt.tz_localize("UTC")
    else:
        combined["timestamp"] = combined["timestamp"].dt.tz_convert("UTC")
    return combined


def _storm_spans(labels: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if labels.empty:
        return []
    sorted_labels = labels.sort_values("timestamp")
    timestamps = sorted_labels["timestamp"]
    step = timestamps.diff().median()
    if pd.isna(step) or step == pd.Timedelta(0):
        step = pd.Timedelta(hours=1)
    storm_flag = sorted_labels["storm_flag"] == 1
    run_id = storm_flag.ne(storm_flag.shift(fill_value=False)).cumsum()
    spans = []
    for run, flag in storm_flag.groupby(run_id):
        if not flag.iloc[0]:
            continue
        run_rows = sorted_labels.loc[run_id == run]
        start = run_rows["timestamp"].iloc[0]
        end = run_rows["timestamp"].iloc[-1] + step
        severity = int(run_rows["storm_severity"].max())
        spans.append((start, end, severity))
    return spans


def plot_kp_year(year: int, output: Path | None = None) -> None:
    dst = _load_dst()
    kp = _load_kp()
    full_storm_labels = _load_full_storm_labels()

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[(dst.index >= start) & (dst.index < end)]
    kp_slice = kp.loc[(kp.index >= start) & (kp.index < end)]
    full_storm_slice = full_storm_labels.loc[
        (full_storm_labels["timestamp"] >= start) & (full_storm_labels["timestamp"] < end)
    ]

    if dst_slice.empty or kp_slice.empty:
        raise ValueError(f"No overlapping DST/Kp data for year {year}.")
    if "storm_flag" not in full_storm_slice.columns or "storm_severity" not in full_storm_slice.columns:
        raise KeyError("Expected full-storm label columns 'storm_flag' and 'storm_severity'.")

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1]
    )
    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], color="tab:blue", label="Dst")
    ax_kp.plot(kp_slice.index, kp_slice["kp"], color="tab:blue", label="Kp index")

    severity_colors = {
        1: "#B7D7F2",  # G1
        2: "#C7E5B3",  # G2
        3: "#F6F2B0",  # G3
        4: "#F6D2A6",  # G4
        5: "#F1B4B4",  # G5
    }

    spans = _storm_spans(full_storm_slice)
    for start, end, severity in spans:
        color = severity_colors.get(severity, "#FFD2A8")
        ax_dst.axvspan(start, end, color=color, alpha=0.35)
        ax_kp.axvspan(start, end, color=color, alpha=0.25)

    kp_peaks = _find_kp_peaks(kp_slice["kp"])

    line_label_used = False
    for peak_time in kp_peaks:
        label = "Kp peak" if not line_label_used else None
        ax_dst.axvline(peak_time, color="#777777", linestyle="--", label=label)
        ax_kp.axvline(peak_time, color="#777777", linestyle="--", label=None)
        line_label_used = True

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")
    ax_dst.axhline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.5)
    ax_dst.axhline(-50, color="black", linewidth=1.5, linestyle=":", alpha=0.3, label="-50 nT")
    ax_kp.axhline(5, color="black", linewidth=1.5, linestyle=":", label="Kp = 5", alpha=0.3)
    ax_dst.set_title(f"DST and Kp index (full-storm windows) in {year}")
    ax_dst.grid(True, alpha=0.3)
    ax_kp.grid(True, alpha=0.3)

    handles_dst, labels_dst = ax_dst.get_legend_handles_labels()
    order_dst = {label: handle for handle, label in zip(handles_dst, labels_dst)}
    legend_entries = [
        ("Dst", order_dst.get("Dst")),
        ("Kp peak", order_dst.get("Kp peak")),
        ("-50 nT", order_dst.get("-50 nT")),
    ]
    for label, color in [
        ("Storm G1", "#B7D7F2"),
        ("Storm G2", "#C7E5B3"),
        ("Storm G3", "#F6F2B0"),
        ("Storm G4", "#F6D2A6"),
        ("Storm G5", "#F1B4B4"),
    ]:
        legend_entries.append(
            (label, Line2D([0], [0], color=color, linewidth=2, alpha=0.85))
        )
    legend_handles = [handle for _, handle in legend_entries if handle is not None]
    legend_labels = [label for label, handle in legend_entries if handle is not None]
    ax_dst.legend(legend_handles, legend_labels)

    handles_kp, labels_kp = ax_kp.get_legend_handles_labels()
    order_kp = {label: handle for handle, label in zip(handles_kp, labels_kp)}
    kp_order = []
    if "Kp index" in order_kp:
        kp_order.append(("Kp index", order_kp["Kp index"]))
    if "Kp = 5" in order_kp:
        kp_order.append(("Kp = 5", order_kp["Kp = 5"]))
    if kp_order:
        ax_kp.legend([h for _, h in kp_order], [lbl for lbl, _ in kp_order])

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
    plot_kp_year(YEAR_TO_PLOT, OUTPUT_PATH)


if __name__ == "__main__":
    main()
