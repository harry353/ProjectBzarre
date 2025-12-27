from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
PREPROCESS_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

DST_DB = PREPROCESS_ROOT / "dst" / "1_averaging" / "dst_aver.db"
KP_DB = PREPROCESS_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"
AP_DB = (
    PREPROCESS_ROOT
    / "kp_index"
    / "5_engineered_features"
    / "kp_index_aver_filt_imp_eng.db"
)
STORM_DB = PREPROCESS_ROOT / "features_targets" / "storm_labels.db"
FINAL_DB = PREPROCESS_ROOT / "final" / "all_sources_intersection.db"

PREDICTIONS_PATH = Path(__file__).resolve().parent / "final_test_predictions.parquet"
YEAR_TO_PLOT = 2024
HORIZON_HOURS = 4

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


def _load_series(table_names: list[str], column: str) -> pd.Series:
    frames = []
    with sqlite3.connect(STORM_DB) as conn:
        for table in table_names:
            df = pd.read_sql_query(
                f"SELECT * FROM {table}",
                conn,
                parse_dates=["timestamp"],
            )
            frames.append(df)
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    combined["timestamp"] = _ensure_utc(combined["timestamp"])
    return combined.set_index("timestamp")[column].astype("Int8")


def _ranges_from_series(series: pd.Series, value_mapper=None):
    ranges = []
    in_range = False
    start = None
    prev_timestamp = None
    for timestamp, value in series.items():
        if value > 0 and not in_range:
            in_range = True
            start = timestamp
        elif value == 0 and in_range:
            end = prev_timestamp if prev_timestamp is not None else timestamp
            label_val = series.loc[start]
            if value_mapper:
                label_val = value_mapper(label_val)
            ranges.append((start, end, label_val))
            in_range = False
            start = None
        prev_timestamp = timestamp
    if in_range and start is not None:
        label_val = series.loc[start]
        if value_mapper:
            label_val = value_mapper(label_val)
        ranges.append((start, prev_timestamp, label_val))
    return ranges


def _load_prediction_timeline() -> pd.Series:
    with sqlite3.connect(FINAL_DB) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, main_phase_label FROM combined_test",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    shifted = df["main_phase_label"].shift(-HORIZON_HOURS)
    valid_mask = shifted.notna()
    timestamps = df.loc[valid_mask, "timestamp"].reset_index(drop=True)
    return timestamps


def _predicted_ranges() -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], pd.Timestamp, pd.Timestamp]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_PATH}")
    preds_df = pd.read_parquet(PREDICTIONS_PATH)
    timestamps = _load_prediction_timeline()
    if len(preds_df) != len(timestamps):
        raise RuntimeError("Prediction rows do not align with timestamp count.")

    predicted_mask = preds_df["final_watch"].to_numpy(dtype=bool)
    ranges = []
    in_range = False
    start = None
    prev_ts = None
    for ts, flag in zip(timestamps, predicted_mask):
        if flag and not in_range:
            in_range = True
            start = ts
        elif not flag and in_range:
            ranges.append((start, prev_ts))
            in_range = False
        prev_ts = ts
    if in_range:
        ranges.append((start, timestamps.iloc[-1]))
    return ranges, timestamps.iloc[0], timestamps.iloc[-1]


def plot_main_phase_predictions(year: int) -> None:
    dst = _load_dst()
    kp = _load_kp()
    ap = _load_ap()
    severity = _load_series(
        ["severity_train", "severity_validation", "severity_test"],
        "severity_label",
    ).reindex(dst.index, fill_value=0)
    main_phase = _load_series(
        ["main_phase_train", "main_phase_validation", "main_phase_test"],
        "main_phase_label",
    ).reindex(dst.index, fill_value=0)

    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = start + pd.DateOffset(years=1)

    dst_slice = dst.loc[(dst.index >= start) & (dst.index < end)]
    kp_slice = kp.loc[(kp.index >= start) & (kp.index < end)]
    severity_slice = severity.loc[(severity.index >= start) & (severity.index < end)]
    main_slice = main_phase.loc[(main_phase.index >= start) & (main_phase.index < end)]

    fig, (ax_dst, ax_kp) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1]
    )
    ax_dst.plot(dst_slice.index, dst_slice["dst_phys"], color="tab:blue", label="Dst")
    ax_kp.plot(kp_slice.index, kp_slice["kp"], color="tab:blue", label="Kp index")

    used_labels: set[str] = set()
    for start_range, end_range, cls in _ranges_from_series(severity_slice):
        info = CLASS_INFO.get(int(cls))
        if not info:
            continue
        grade, color = info
        label = grade if grade not in used_labels else None
        ax_dst.axvspan(start_range, end_range, color=color, alpha=0.2, label=label)
        used_labels.add(grade)

    actual_main_ranges = _ranges_from_series(main_slice)
    main_phase_label_used = False
    for m_start, m_end, cls in actual_main_ranges:
        info = CLASS_INFO.get(int(cls))
        if not info:
            continue
        _, color = info
        label = "Actual main phase" if not main_phase_label_used else None
        ax_dst.axvspan(m_start, m_end, color=color, alpha=0.35, label=label)
        if label:
            main_phase_label_used = True

    predicted_ranges, pred_start, pred_end = _predicted_ranges()
    predicted_label_used = False
    drew_prediction = False
    for p_start, p_end in predicted_ranges:
        if p_end < start or p_start > end:
            continue
        label = "Predicted main phase" if not predicted_label_used else None
        ax_dst.axvspan(
            max(p_start, start),
            min(p_end, end),
            color="#800080",
            alpha=0.25,
            label=label,
        )
        predicted_label_used = True
        drew_prediction = True

    if not drew_prediction:
        print(
            f"[WARN] No predictions intersect with year {year}. "
            f"Predictions cover {pred_start:%Y-%m-%d} to {pred_end:%Y-%m-%d}."
        )

    ax_dst.set_ylabel("Dst (nT)")
    ax_kp.set_ylabel("Kp")
    ax_kp.set_xlabel("Time")
    ax_dst.axhline(0, color="black", linewidth=1.5, linestyle="-", alpha=0.5)
    ax_dst.axhline(-50, color="black", linewidth=1.5, linestyle=":", alpha=0.3)
    ax_kp.axhline(5, color="black", linewidth=1.5, linestyle=":", alpha=0.3)
    ax_dst.set_title(f"DST, Kp, and predicted main phases in {year}")
    ax_dst.grid(True, alpha=0.3)
    ax_kp.grid(True, alpha=0.3)

    handles_dst, labels_dst = ax_dst.get_legend_handles_labels()
    ax_dst.legend(handles_dst, labels_dst, loc="upper right")

    handles_kp, labels_kp = ax_kp.get_legend_handles_labels()
    ax_kp.legend(handles_kp, labels_kp, loc="upper right")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_main_phase_predictions(YEAR_TO_PLOT)
