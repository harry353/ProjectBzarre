from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_DIR
for parent in STAGE_DIR.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = STAGE_DIR.parent

DST_DB = PROJECT_ROOT / "preprocessing_pipeline" / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "kp_index"
    / "1_averaging"
    / "kp_index_aver.db"
)
AP_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "kp_index"
    / "5_engineered_features"
    / "kp_index_aver_filt_imp_eng.db"
)
DAILY_DB = STAGE_DIR / "storm_daily_labels.db"
DAILY_TABLES = [
    "storm_daily_train",
    "storm_daily_validation",
    "storm_daily_test",
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


def _load_daily_labels() -> pd.DataFrame:
    frames = []
    with sqlite3.connect(DAILY_DB) as conn:
        for table in DAILY_TABLES:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn, parse_dates=["date"])
                frames.append(df)
    if not frames:
        raise RuntimeError("Daily storm labels not found.")
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
    )
    combined["date"] = pd.to_datetime(combined["date"])
    if combined["date"].dt.tz is None:
        combined["date"] = combined["date"].dt.tz_localize("UTC")
    else:
        combined["date"] = combined["date"].dt.tz_convert("UTC")
    return combined


def plot_ap_year(year: int, output: Path | None = None) -> None:
    dst = _load_dst()
    kp = _load_kp()
    ap = _load_ap()
    daily_labels = _load_daily_labels()

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[(dst.index >= start) & (dst.index < end)]
    kp_slice = kp.loc[(kp.index >= start) & (kp.index < end)]
    ap_slice = ap.loc[(ap.index >= start) & (ap.index < end)]
    daily_slice = daily_labels.loc[(daily_labels["date"] >= start) & (daily_labels["date"] < end)]

    if dst_slice.empty or kp_slice.empty:
        raise ValueError(f"No overlapping DST/Kp data for year {year}.")

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1]
    )
    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], color="tab:blue", label="Dst")
    ax_kp.plot(kp_slice.index, kp_slice["kp"], color="tab:blue", label="Kp index")

    storm_label_col = "storm_severity_next_8h"
    severity_colors = {
        1: "#B7D7F2",  # G1
        2: "#C7E5B3",  # G2
        3: "#F6F2B0",  # G3
        4: "#F6D2A6",  # G4
        5: "#F1B4B4",  # G5
    }
    if storm_label_col not in daily_slice.columns:
        raise KeyError(f"Expected daily label column '{storm_label_col}' not found.")

    storm_label_used = False
    for _, row in daily_slice.iterrows():
        severity = int(row[storm_label_col])
        if severity <= 0:
            continue
        day_start = row["date"]
        day_end = day_start + pd.Timedelta(hours=8)
        label = None
        ax_dst.axvspan(
            day_start,
            day_end,
            color=severity_colors.get(severity, "#FFD2A8"),
            alpha=0.35,
            label=label,
        )
        storm_label_used = True

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
    ax_dst.axhline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.5)
    ax_dst.axhline(-50, color="black", linewidth=1.5, linestyle=":", alpha=0.3, label="-50 nT")
    ax_kp.axhline(5, color="black", linewidth=1.5, linestyle=":", label="Kp = 5", alpha=0.3)
    ax_dst.set_title(f"DST and Kp index (Ap-driven targets) in {year}")
    ax_dst.grid(True, alpha=0.3)
    ax_kp.grid(True, alpha=0.3)
    legend_order = [
        "Dst",
        "Ap peak",
        "-50 nT",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
    ]
    handles_dst, labels_dst = ax_dst.get_legend_handles_labels()
    order_dst = {label: handle for handle, label in zip(handles_dst, labels_dst)}
    for label, color in [
        ("G1", "#B7D7F2"),
        ("G2", "#C7E5B3"),
        ("G3", "#F6F2B0"),
        ("G4", "#F6D2A6"),
        ("G5", "#F1B4B4"),
    ]:
        order_dst[label] = plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.35)
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
