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

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

IMF_DB = (
    PIPELINE_ROOT
    / "imf_solar_wind"
    / "6_engineered_features"
    / "imf_solar_wind_aver_comb_filt_imp_eng.db"
)
IMF_TABLE = "engineered_features"

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"

LABELS_DB = STAGE_DIR / "disturbed_labels.db"
LABEL_TABLES = [
    "disturbed_train",
    "disturbed_validation",
    "disturbed_test",
]

YEAR_TO_PLOT = 2024
OUTPUT_PATH: Path | None = None


def _ensure_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize("UTC")
        if series.dt.tz is None
        else series.dt.tz_convert("UTC")
    )


def _load_table(db: Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp").sort_index()


def _load_imf() -> pd.DataFrame:
    return _load_table(
        IMF_DB,
        "SELECT time_tag AS timestamp, bz_gse, newell_dphi_dt, speed, dynamic_pressure "
        f"FROM {IMF_TABLE}",
    )


def _load_dst() -> pd.DataFrame:
    return _load_table(
        DST_DB,
        "SELECT time_tag AS timestamp, dst FROM hourly_data",
    )


def _load_kp() -> pd.DataFrame:
    return _load_table(
        KP_DB,
        "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
    )


def _load_labels() -> pd.DataFrame:
    frames = []
    with sqlite3.connect(LABELS_DB) as conn:
        for table in LABEL_TABLES:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                frame = pd.read_sql_query(
                    f"SELECT time_tag AS timestamp, disturbed_flag FROM {table}",
                    conn,
                    parse_dates=["timestamp"],
                )
                frames.append(frame)
    if not frames:
        raise RuntimeError("Disturbed labels not found.")
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
    return combined.set_index("timestamp").sort_index()


def _disturbance_spans(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if labels.empty:
        return []
    active = labels == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()
    spans = []
    for run, flag in active.groupby(run_id):
        if not flag.iloc[0]:
            continue
        run_index = labels.index[run_id == run]
        spans.append((run_index[0], run_index[-1]))
    return spans


def plot_disturbed_labels(year: int, output: Path | None = None) -> None:
    imf = _load_imf()
    dst = _load_dst()
    kp = _load_kp()
    labels = _load_labels()

    common_index = imf.index.intersection(dst.index).intersection(kp.index)
    combined = (
        imf.loc[common_index]
        .join(dst.loc[common_index])
        .join(kp.loc[common_index])
        .join(labels)
        .dropna(subset=["disturbed_flag"])
    )

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)
    combined = combined.loc[(combined.index >= start) & (combined.index < end)]

    if combined.empty:
        raise ValueError(f"No aligned data for year {year}.")

    spans = _disturbance_spans(combined["disturbed_flag"])

    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(combined.index, combined["bz_gse"], color="#1f77b4", linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Bz (nT)")

    axes[1].plot(combined.index, combined["newell_dphi_dt"], color="#ff7f0e", linewidth=0.8)
    axes[1].set_ylabel("Newell dPhi/dt")

    axes[2].plot(combined.index, combined["speed"], color="#d62728", linewidth=0.8)
    axes[2].set_ylabel("Speed (km/s)")

    axes[3].plot(
        combined.index, combined["dynamic_pressure"], color="#9467bd", linewidth=0.8
    )
    axes[3].set_ylabel("Dyn. Pressure")

    axes[4].plot(combined.index, combined["dst"], color="#8c564b", linewidth=0.8)
    axes[4].axhline(-50.0, color="black", linewidth=0.8, alpha=0.4)
    axes[4].set_ylabel("Dst (nT)")

    axes[5].plot(combined.index, combined["kp_index"], color="#7f7f7f", linewidth=0.8)
    axes[5].axhline(5.0, color="black", linewidth=0.8, alpha=0.4)
    axes[5].set_ylabel("Kp")
    axes[5].set_xlabel("Time (UTC)")

    for ax in axes:
        for start_time, end_time in spans:
            ax.axvspan(start_time, end_time, color="#f1c40f", alpha=0.2, zorder=0)
        ax.grid(True, alpha=0.3, linewidth=0.6)

    fig.suptitle(f"Disturbed labels in {year}", y=0.995)
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"[OK] Figure saved to {output}")
    else:
        plt.show()
        plt.close(fig)


def main() -> None:
    plot_disturbed_labels(YEAR_TO_PLOT, OUTPUT_PATH)


if __name__ == "__main__":
    main()
