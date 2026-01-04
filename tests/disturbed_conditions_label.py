from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMF_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "imf_solar_wind"
    / "6_engineered_features"
    / "imf_solar_wind_aver_comb_filt_imp_eng.db"
)
IMF_TABLE = "engineered_features"

DST_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "dst"
    / "1_averaging"
    / "dst_aver.db"
)
DST_TABLE = "hourly_data"

KP_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "kp_index"
    / "1_averaging"
    / "kp_index_aver.db"
)
KP_TABLE = "hourly_data"

OUTPUT_FIG = Path(__file__).resolve().with_suffix(".png")
YEAR_TO_PLOT = 2024
REQUIRED_CONDITIONS = 2


def _load_table(db_path: Path, table: str, columns: list[str]) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT time_tag, {', '.join(columns)} FROM {table}",
            conn,
            parse_dates=["time_tag"],
        )
    df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
    df = df.dropna(subset=["time_tag"]).set_index("time_tag").sort_index()
    return df


def _load_aligned_data() -> pd.DataFrame:
    imf_cols = ["bz_gse", "bt", "newell_dphi_dt", "epsilon", "speed", "dynamic_pressure"]
    imf = _load_table(IMF_DB, IMF_TABLE, imf_cols)
    dst = _load_table(DST_DB, DST_TABLE, ["dst"])
    kp = _load_table(KP_DB, KP_TABLE, ["kp_index"])
    return imf.join([dst, kp], how="inner")


def plot_disturbed_conditions() -> None:
    df = _load_aligned_data()
    if YEAR_TO_PLOT is not None:
        df = df[df.index.year == YEAR_TO_PLOT]

    theta1 = 2.0e4
    theta2 = 4.0
    bz_south = df["bz_gse"] <= -6.0
    southward_flag = bz_south.rolling(2, min_periods=2).sum() >= 2
    strong_coupling = df["newell_dphi_dt"].rolling(3, min_periods=3).sum() > theta1
    high_speed = (df["speed"] >= 600.0).rolling(2, min_periods=2).sum() >= 2
    compression = df["dynamic_pressure"].diff() > theta2
    condition_count = (
        southward_flag.astype(int)
        + strong_coupling.astype(int)
        + high_speed.astype(int)
        + compression.astype(int)
    )
    disturbed_flag = (condition_count >= REQUIRED_CONDITIONS).fillna(False)
    disturbed_pct = disturbed_flag.mean() * 100.0
    print(f"Disturbed percentage: {disturbed_pct:.2f}%")
    kp_values = df["kp_index"].to_numpy()
    peak_indices, props = find_peaks(kp_values, height=5.0)
    peak_times = df.index[peak_indices]

    starts = disturbed_flag & ~disturbed_flag.shift(1, fill_value=False)
    ends = ~disturbed_flag & disturbed_flag.shift(1, fill_value=False)
    start_times = df.index[starts]
    end_times = df.index[ends]
    if disturbed_flag.iloc[-1]:
        end_times = end_times.append(pd.Index([df.index[-1]]))
    intervals = list(zip(start_times, end_times))

    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(df.index, df["bz_gse"], color="#1f77b4", linewidth=0.8, label="Bz")
    axes[0].plot(df.index, df["bt"], color="#17becf", linewidth=0.8, label="Bt")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_ylabel("Bz (nT)")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)

    axes[1].plot(df.index, df["newell_dphi_dt"], color="#ff7f0e", linewidth=0.8)
    axes[1].set_ylabel("Newell dPhi/dt (kV/s)")

    axes[2].plot(df.index, df["epsilon"], color="#2ca02c", linewidth=0.8)
    axes[2].set_ylabel("Epsilon (W)")

    axes[3].plot(df.index, df["speed"], color="#d62728", linewidth=0.8)
    axes[3].set_ylabel("Speed (km/s)")

    axes[4].plot(df.index, df["dynamic_pressure"], color="#9467bd", linewidth=0.8)
    axes[4].set_ylabel("Dyn. Pressure (nPa)")

    axes[5].plot(df.index, df["dst"], color="#8c564b", linewidth=0.8)
    axes[5].axhline(-50.0, color="black", linewidth=0.8, alpha=0.6)
    axes[5].set_ylabel("Dst (nT)")

    axes[6].plot(df.index, df["kp_index"], color="#7f7f7f", linewidth=0.8)
    axes[6].axhline(5.0, color="black", linewidth=0.8, alpha=0.6)
    axes[6].set_ylabel("Kp (index)")
    axes[6].set_xlabel("Time (UTC)")

    for ax in axes:
        for start, end in intervals:
            ax.axvspan(start, end, color="#f1c40f", alpha=0.2, zorder=0)
        for peak_time in peak_times:
            ax.axvline(peak_time, color="#4d4d4d", linewidth=0.9, alpha=0.6)
        ax.grid(True, alpha=0.3, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=150)
    plt.show()


def main() -> None:
    plot_disturbed_conditions()


if __name__ == "__main__":
    main()
