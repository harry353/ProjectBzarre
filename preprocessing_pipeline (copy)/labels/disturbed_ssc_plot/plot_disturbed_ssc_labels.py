from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

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

DISTURBED_DB = PIPELINE_ROOT / "features_targets" / "disturbed_label" / "disturbed_labels.db"
DISTURBED_TABLES = [
    "disturbed_train",
    "disturbed_validation",
    "disturbed_test",
]

SSC_DB = PIPELINE_ROOT / "features_targets" / "full_storm_label" / "full_storm_labels.db"
SSC_TABLES = [
    "storm_full_storm_train",
    "storm_full_storm_validation",
    "storm_full_storm_test",
]

YEAR_TO_PLOT = 2024
# LEAD_HOURS_LIST = [6, 8, 12]
LEAD_HOURS_LIST = [12]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

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


def _load_disturbed_labels() -> pd.DataFrame:
    frames = []
    with sqlite3.connect(DISTURBED_DB) as conn:
        for table in DISTURBED_TABLES:
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

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = _ensure_utc(pd.to_datetime(combined["timestamp"]))
    return combined.set_index("timestamp").sort_index()


def _load_full_storm_labels() -> pd.DataFrame:
    frames = []
    with sqlite3.connect(SSC_DB) as conn:
        for table in SSC_TABLES:
            if not pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=(table,),
            ).empty:
                frame = pd.read_sql_query(
                    f"SELECT timestamp, storm_flag, storm_severity FROM {table}",
                    conn,
                    parse_dates=["timestamp"],
                )
                frames.append(frame)

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = _ensure_utc(pd.to_datetime(combined["timestamp"]))
    return combined.set_index("timestamp").sort_index()


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def stage1_recall(
    disturbed: pd.Series,
    storm_onsets: list[pd.Timestamp],
    lead_hours: int,
) -> float:
    hits = 0

    for onset in storm_onsets:
        window_start = onset - pd.Timedelta(hours=lead_hours)
        window = disturbed.loc[
            (disturbed.index >= window_start)
            & (disturbed.index < onset)
        ]

        if (window == 1).any():
            hits += 1

    return hits / len(storm_onsets) if storm_onsets else float("nan")


def stage1_precision(
    disturbed: pd.Series,
    storm_onsets: list[pd.Timestamp],
    lead_hours: int,
) -> float:
    if disturbed.empty:
        return float("nan")

    hits = 0
    total = int((disturbed == 1).sum())

    if total == 0:
        return float("nan")

    storm_onsets = pd.Series(storm_onsets)

    for t in disturbed.index[disturbed == 1]:
        future_window_end = t + pd.Timedelta(hours=lead_hours)
        if ((storm_onsets > t) & (storm_onsets <= future_window_end)).any():
            hits += 1

    return hits / total


def disturbance_spans(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    active = labels == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()
    spans = []
    for run, flag in active.groupby(run_id):
        if flag.iloc[0]:
            idx = labels.index[run_id == run]
            spans.append((idx[0], idx[-1]))
    return spans


def storm_spans(labels: pd.DataFrame) -> list[tuple[pd.Timestamp, int]]:
    active = labels["storm_flag"] == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()
    spans = []
    for run, flag in active.groupby(run_id):
        if flag.iloc[0]:
            rows = labels.loc[run_id == run]
            spans.append((rows.index[0], int(rows["storm_severity"].max())))
    return spans


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    dst = _load_dst()
    kp = _load_kp()
    disturbed = _load_disturbed_labels()
    full_storm = _load_full_storm_labels()

    years = sorted(
        set(disturbed.index.year).intersection(set(full_storm.index.year))
    )

    def _year_metrics(year: int) -> tuple[int, dict[int, tuple[float, float]]]:
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        end = start + pd.DateOffset(years=1)
        disturbed_year = disturbed.loc[start:end, "disturbed_flag"]
        full_storm_year = full_storm.loc[start:end]
        storm_onsets = [t for t, _ in storm_spans(full_storm_year)]
        per_lead = {}
        for h in LEAD_HOURS_LIST:
            r = stage1_recall(disturbed_year, storm_onsets, h)
            p = stage1_precision(disturbed_year, storm_onsets, h)
            per_lead[h] = (r, p)
        return year, per_lead

    year_results: list[tuple[int, dict[int, tuple[float, float]]]] = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_year_metrics, year): year for year in years}
        for future in as_completed(futures):
            year_results.append(future.result())

    year_results.sort(key=lambda x: x[0])
    print("Stage-1 metrics (per year):")
    for year, per_lead in year_results:
        parts = []
        for h in LEAD_HOURS_LIST:
            r, p = per_lead[h]
            parts.append(f"{h}h r={r:.3f} p={p:.3f}")
        print(f"{year} | " + " | ".join(parts))

    avg_parts = []
    for h in LEAD_HOURS_LIST:
        recalls = [
            per_lead[h][0]
            for _, per_lead in year_results
            if not pd.isna(per_lead[h][0])
        ]
        precisions = [
            per_lead[h][1]
            for _, per_lead in year_results
            if not pd.isna(per_lead[h][1])
        ]
        mean_recall = float(pd.Series(recalls).mean()) if recalls else float("nan")
        mean_precision = float(pd.Series(precisions).mean()) if precisions else float("nan")
        avg_parts.append(f"{h}h r={mean_recall:.3f} p={mean_precision:.3f}")
    print("AVG | " + " | ".join(avg_parts))

    start = pd.Timestamp(year=YEAR_TO_PLOT, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst = dst.loc[start:end]
    kp = kp.loc[start:end]
    disturbed = disturbed.loc[start:end]
    full_storm = full_storm.loc[start:end]

    disturbed_series = disturbed["disturbed_flag"]
    storm_spans_list = storm_spans(full_storm)
    storm_onsets = [t for t, _ in storm_spans_list]

    # ---------------- Plot ----------------

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax_dst.plot(dst.index, dst["dst"], color="tab:blue")
    ax_kp.plot(kp.index, kp["kp_index"], color="tab:gray")

    for s, e in disturbance_spans(disturbed_series):
        ax_dst.axvspan(s, e, color="#f1c40f", alpha=0.2)
        ax_kp.axvspan(s, e, color="#f1c40f", alpha=0.2)

    severity_colors = {
        1: "#B7D7F2",
        2: "#C7E5B3",
        3: "#F6F2B0",
        4: "#F6D2A6",
        5: "#F1B4B4",
    }

    for onset, sev in storm_spans_list:
        ax_dst.axvspan(
            onset,
            onset + pd.Timedelta(hours=1),
            color=severity_colors.get(sev, "#FFD2A8"),
            alpha=0.35,
        )

    ax_dst.axhline(-50, linestyle=":", color="black", alpha=0.4)
    ax_kp.axhline(5, linestyle=":", color="black", alpha=0.4)

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")

    ax_dst.set_title(f"Disturbed gate and SSC storms ({YEAR_TO_PLOT})")

    legend = [
        Patch(facecolor="#f1c40f", alpha=0.2, label="Disturbed"),
        Patch(facecolor=severity_colors[3], alpha=0.35, label="Storm"),
    ]
    ax_dst.legend(handles=legend, frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
