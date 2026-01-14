from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
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

OUTPUT_DB = STAGE_DIR / "full_storm_labels.db"
TRAIN_TABLE = "storm_full_storm_train"
VAL_TABLE = "storm_full_storm_validation"
TEST_TABLE = "storm_full_storm_test"

PEAK_PROMINENCE = 39.0
MAX_PRE_PEAK_OFFSET = pd.Timedelta(hours=24)
STORM_OVERLAP_ALLOWANCE = pd.Timedelta(hours=4)
FORECAST_HORIZONS_H = range(1, 9)

TRAIN_START = pd.Timestamp("1999-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2016-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2017-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2020-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2021-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-11-30T23:59:59Z")

GRADE_TO_CLASS = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
KP_TO_AP = {
    0.00: 0,
    0.33: 2,
    0.67: 3,
    1.00: 4,
    1.33: 5,
    1.67: 6,
    2.00: 7,
    2.33: 9,
    2.67: 12,
    3.00: 15,
    3.33: 18,
    3.67: 22,
    4.00: 27,
    4.33: 32,
    4.67: 39,
    5.00: 48,
    5.33: 56,
    5.67: 67,
    6.00: 80,
    6.33: 94,
    6.67: 111,
    7.00: 132,
    7.33: 154,
    7.67: 179,
    8.00: 207,
    8.33: 236,
    8.67: 300,
    9.00: 400,
}
KP_KEYS = np.array(sorted(KP_TO_AP.keys()), dtype=float)


def _ensure_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize("UTC")
        if series.dt.tz is None
        else series.dt.tz_convert("UTC")
    )


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, dst FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["dst"]


def _load_kp() -> pd.Series:
    with sqlite3.connect(KP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["kp_index"]


def _kp_to_ap(kp_series: pd.Series) -> pd.Series:
    kp_values = kp_series.astype(float).round(2)
    mapped = kp_values.map(KP_TO_AP)
    missing = mapped.isna()
    if missing.any():
        values = kp_values[missing].to_numpy()
        idx = np.abs(values[:, None] - KP_KEYS[None, :]).argmin(axis=1)
        nearest = KP_KEYS[idx]
        mapped.loc[missing] = [KP_TO_AP[key] for key in nearest]
    return mapped.astype(float)


def _find_peaks(series: pd.Series) -> list[pd.Timestamp]:
    values = series.to_numpy()
    peaks, _ = find_peaks(values, prominence=PEAK_PROMINENCE)
    return [series.index[idx] for idx in peaks]


def _classify_kp(kp_value: float) -> str | None:
    if kp_value >= 9:
        return "G5"
    if kp_value >= 8:
        return "G4"
    if kp_value >= 7:
        return "G3"
    if kp_value >= 6:
        return "G2"
    if kp_value >= 5:
        return "G1"
    return None


def _compute_ranges(
    dst_series: pd.Series,
    ap_series: pd.Series,
    kp_series: pd.Series,
    peaks: list[pd.Timestamp],
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for peak_time in peaks:
        past = dst_series[dst_series.index < peak_time]
        start = None
        if not past.empty:
            last_positive = past[past > 0].dropna()
            if not last_positive.empty:
                start = last_positive.index[-1]
        if start is None:
            start = peak_time
        if peak_time - start > MAX_PRE_PEAK_OFFSET:
            start = peak_time - MAX_PRE_PEAK_OFFSET

        future = dst_series[dst_series.index > peak_time]
        if future.empty:
            continue

        if dst_series.get(peak_time, 0) >= 0:
            crossing = future[future <= 0]
            if crossing.empty:
                continue
            start = crossing.index[0]
            future = dst_series[dst_series.index > start]
            if future.empty:
                continue

        post_start = dst_series[dst_series.index >= start]
        non_positive_start = post_start[post_start <= 0]
        if non_positive_start.empty:
            continue
        start = non_positive_start.index[0]

        non_positive = future[future <= 0]
        if non_positive.empty:
            continue
        crossing = future[future > 0]
        if crossing.empty:
            continue
        first_positive_idx = crossing.index[0]
        end_candidates = dst_series[
            (dst_series.index <= first_positive_idx) & (dst_series <= 0)
        ]
        if end_candidates.empty:
            continue
        end = end_candidates.index[-1]
        if end <= start:
            continue

        ap_window = ap_series[(ap_series.index >= start) & (ap_series.index <= end)]
        kp_window = kp_series[(kp_series.index >= start) & (kp_series.index <= end)]
        if ap_window.empty or kp_window.empty:
            continue
        if dst_series[(dst_series.index >= start) & (dst_series.index <= end)].min() > -50:
            continue
        severity = _classify_kp(float(kp_window.max()))
        if severity is None:
            continue
        ranges.append((start, end, severity))
    return ranges


def _deduplicate_ranges(
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]]
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    collapsed: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for start, end, grade in ranges:
        overlap = False
        for c_start, c_end, _ in collapsed:
            overlap_duration = min(end, c_end) - max(start, c_start)
            if overlap_duration >= STORM_OVERLAP_ALLOWANCE:
                overlap = True
                break
        if not overlap:
            collapsed.append((start, end, grade))
    return collapsed


def _build_storm_labels(
    index: pd.DatetimeIndex, ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]]
) -> pd.DataFrame:
    labels = pd.DataFrame(index=index)
    labels["storm_flag"] = 0
    labels["storm_severity"] = 0
    for start, end, grade in ranges:
        cls = GRADE_TO_CLASS.get(grade, 0)
        if cls == 0:
            continue
        mask = (labels.index >= start) & (labels.index <= end)
        if not mask.any():
            continue
        labels.loc[mask, "storm_flag"] = 1
        existing = labels.loc[mask, "storm_severity"]
        labels.loc[mask, "storm_severity"] = existing.where(existing >= cls, cls)
    labels = labels.reset_index().rename(columns={"index": "timestamp"})
    return labels


def _add_forecast_horizons(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    for h in FORECAST_HORIZONS_H:
        out[f"storm_flag_h{h}"] = out["storm_flag"].shift(-h)
        out[f"storm_severity_h{h}"] = out["storm_severity"].shift(-h)
    horizon_cols = [
        f"storm_flag_h{h}" for h in FORECAST_HORIZONS_H
    ] + [
        f"storm_severity_h{h}" for h in FORECAST_HORIZONS_H
    ]
    out = out.dropna(subset=horizon_cols)
    return out


def _split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[mask].copy()


def build_full_storm_labels() -> None:
    dst = _load_dst()
    kp = _load_kp()
    ap = _kp_to_ap(kp)

    common_index = dst.index.intersection(ap.index).intersection(kp.index)
    dst = dst.loc[common_index]
    ap = ap.loc[common_index]
    kp = kp.loc[common_index]

    peaks = _find_peaks(ap)
    ranges = _deduplicate_ranges(_compute_ranges(dst, ap, kp, peaks))

    labels = _build_storm_labels(dst.index, ranges)
    labels = _add_forecast_horizons(labels)
    train = _split(labels, TRAIN_START, TRAIN_END)
    val = _split(labels, VAL_START, VAL_END)
    test = _split(labels, TEST_START, TEST_END)

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    storm_pct = labels["storm_flag"].mean() * 100.0
    print(f"[OK] Full-storm labels written to {OUTPUT_DB}")
    print(f"     Train / Val / Test rows: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"     Storm coverage percentage: {storm_pct:.2f}%")


def main() -> None:
    build_full_storm_labels()


if __name__ == "__main__":
    main()
