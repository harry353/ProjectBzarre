from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent
DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"
AP_DB = (
    PIPELINE_ROOT
    / "kp_index"
    / "5_engineered_features"
    / "kp_index_aver_filt_imp_eng.db"
)
STORM_DB = STAGE_DIR / "storm_labels.db"
SEVERITY_TABLES = ["severity_train", "severity_validation", "severity_test"]
SSC_TABLES = ["ssc_train", "ssc_validation", "ssc_test"]
MAIN_PHASE_TABLES = ["main_phase_train", "main_phase_validation", "main_phase_test"]

YEAR_TO_PLOT = 2003
OUTPUT_PATH: Path | None = None

CLASS_INFO = {
    1: ("G1", "#FFFF00"),
    2: ("G2", "#FFD200"),
    3: ("G3", "#FF8C00"),
    4: ("G4", "#FF0000"),
    5: ("G5", "#8B0000"),
}


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


def _load_ap() -> pd.Series:
    with sqlite3.connect(AP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, ap FROM engineered_features",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["ap"]


def _find_ap_peaks(series: pd.Series) -> list[pd.Timestamp]:
    values = series.to_numpy()
    peaks, _ = find_peaks(values, prominence=39.0)
    return [series.index[idx] for idx in peaks]


def _load_severity() -> pd.Series:
    frames = []
    with sqlite3.connect(STORM_DB) as conn:
        for table in SEVERITY_TABLES:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table}",
                    conn,
                    parse_dates=["timestamp"],
                )
                frames.append(df)
    if not frames:
        raise RuntimeError("Severity labels not found in storm_labels.db")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = _ensure_utc(combined["timestamp"])
    return combined.set_index("timestamp")["severity_label"].astype("Int8")


def _load_binary_label(tables: list[str], column: str) -> pd.Series:
    frames = []
    with sqlite3.connect(STORM_DB) as conn:
        for table in tables:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                df = pd.read_sql_query(
                    f"SELECT * FROM {table}",
                    conn,
                    parse_dates=["timestamp"],
                )
                frames.append(df)
    if not frames:
        raise RuntimeError(f"Label '{column}' not found in storm_labels.db")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = _ensure_utc(combined["timestamp"])
    return combined.set_index("timestamp")[column].astype("Int8")


def _severity_ranges(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    current_class = 0
    start = None
    prev_timestamp = None
    for timestamp, value in series.items():
        if value > 0 and current_class == 0:
            start = timestamp
            current_class = int(value)
        elif value != current_class and current_class > 0:
            end = prev_timestamp if prev_timestamp is not None else timestamp
            ranges.append((start, end, current_class))
            current_class = int(value)
            start = timestamp if value > 0 else None
        prev_timestamp = timestamp
    if current_class > 0 and start is not None:
        ranges.append((start, prev_timestamp, current_class))
    return ranges


def _binary_ranges(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_range = False
    start = None
    prev_timestamp = None
    for timestamp, value in series.items():
        if value > 0 and not in_range:
            in_range = True
            start = timestamp
        elif value == 0 and in_range:
            end = prev_timestamp if prev_timestamp is not None else timestamp
            ranges.append((start, end))
            in_range = False
            start = None
        prev_timestamp = timestamp
    if in_range and start is not None:
        ranges.append((start, prev_timestamp))
    return ranges


def plot_ap_year(year: int, output: Path | None = None) -> None:
    dst = _load_dst()
    kp = _load_kp()
    ap = _load_ap()
    severity = _load_severity().reindex(dst.index, fill_value=0)
    ssc = _load_binary_label(SSC_TABLES, "ssc_label").reindex(dst.index, fill_value=0)
    main_phase = _load_binary_label(MAIN_PHASE_TABLES, "main_phase_label").reindex(dst.index, fill_value=0)

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[(dst.index >= start) & (dst.index < end)]
    kp_slice = kp.loc[(kp.index >= start) & (kp.index < end)]
    ap_slice = ap.loc[(ap.index >= start) & (ap.index < end)]
    severity_slice = severity.loc[(severity.index >= start) & (severity.index < end)]
    ssc_slice = ssc.loc[(ssc.index >= start) & (ssc.index < end)]
    main_slice = main_phase.loc[(main_phase.index >= start) & (main_phase.index < end)]

    if dst_slice.empty or kp_slice.empty or severity_slice.empty:
        raise ValueError(f"No overlapping DST/Kp severity data for year {year}.")

    raw_ranges = _severity_ranges(severity_slice)

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1]
    )
    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], color="tab:blue", label="Dst")
    ax_kp.plot(kp_slice.index, kp_slice["kp"], color="tab:blue", label="Kp index")

    used_labels: set[str] = set()
    main_phase_label_used = False
    positive_label_used = False
    for start_range, end_range, cls in raw_ranges:
        info = CLASS_INFO.get(cls)
        if not info:
            continue
        grade, color = info
        label = grade if grade not in used_labels else None
        ax_dst.axvspan(start_range, end_range, color=color, alpha=0.2, label=label)
        used_labels.add(grade)

    ssc_ranges = _binary_ranges(ssc_slice)
    for ssc_start, ssc_end in ssc_ranges:
        label = "SSC" if not positive_label_used else None
        ax_dst.axvspan(ssc_start, ssc_end, color="#008000", alpha=0.15, label=label)
        positive_label_used = True

    main_ranges = _binary_ranges(main_slice)
    for m_start, m_end in main_ranges:
        # Use severity color if available
        severity_window = severity_slice.loc[(severity_slice.index >= m_start) & (severity_slice.index <= m_end)]
        cls = int(severity_window.iloc[0]) if not severity_window.empty else 0
        if cls == 0:
            continue
        grade_info = CLASS_INFO.get(cls)
        if not grade_info:
            continue
        _, color = grade_info
        label = "Main phase" if not main_phase_label_used else None
        ax_dst.axvspan(m_start, m_end, color=color, alpha=0.2, label=label)
        if label:
            main_phase_label_used = True

    ap_peaks = _find_ap_peaks(ap_slice)

    line_label_used = False
    for peak_time in ap_peaks:
        label = "Ap peak" if not line_label_used else None
        ax_dst.axvline(peak_time, color="#777777", linestyle="--", label=label)
        ax_kp.axvline(peak_time, color="#777777", linestyle="--", label=None)
        line_label_used = True

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")
    ax_dst.axhline(0, color="black", linewidth=1.5, linestyle="-", label="-50 nT", alpha=0.5)
    ax_dst.axhline(-50, color="black", linewidth=1.5, linestyle=":", label="-50 nT", alpha=0.3)
    ax_kp.axhline(5, color="black", linewidth=1.5, linestyle=":", label="Kp = 5", alpha=0.3)
    ax_dst.set_title(f"DST and Kp index (Ap-driven targets) in {year}")
    ax_dst.grid(True, alpha=0.3)
    ax_kp.grid(True, alpha=0.3)
    legend_order = ["Dst", "SSC", "Main phase", "Kp index", "G1", "G2", "G3", "G4", "G5"]
    handles_dst, labels_dst = ax_dst.get_legend_handles_labels()
    order_dst = {label: handle for handle, label in zip(handles_dst, labels_dst)}
    ordered_dst_handles = [
        order_dst[label] for label in legend_order if label in order_dst
    ]
    ax_dst.legend(ordered_dst_handles, [label for label in legend_order if label in order_dst])

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
    plot_ap_year(YEAR_TO_PLOT, OUTPUT_PATH)


if __name__ == "__main__":
    main()
