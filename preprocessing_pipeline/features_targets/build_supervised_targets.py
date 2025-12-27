from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from scipy.signal import find_peaks

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = STAGE_DIR.parent

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
AP_DB = (
    PIPELINE_ROOT
    / "kp_index"
    / "5_engineered_features"
    / "kp_index_aver_filt_imp_eng.db"
)
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"

OUTPUT_DB = STAGE_DIR / "storm_labels.db"
TRAIN_TABLE = "severity_train"
VAL_TABLE = "severity_validation"
TEST_TABLE = "severity_test"
SSC_TRAIN_TABLE = "ssc_train"
SSC_VAL_TABLE = "ssc_validation"
SSC_TEST_TABLE = "ssc_test"
MAIN_TRAIN_TABLE = "main_phase_train"
MAIN_VAL_TABLE = "main_phase_validation"
MAIN_TEST_TABLE = "main_phase_test"

PEAK_PROMINENCE = 39.0
PEAK_WINDOW_HOURS = 6
MAX_PRE_PEAK_OFFSET = pd.Timedelta(hours=24)
STORM_OVERLAP_ALLOWANCE = pd.Timedelta(hours=4)

TRAIN_START = pd.Timestamp("1999-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2016-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2017-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2020-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2021-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-11-30T23:59:59Z")

GRADE_TO_CLASS = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}


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


def _load_ap() -> pd.Series:
    with sqlite3.connect(AP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, ap FROM engineered_features",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["ap"]


def _load_kp() -> pd.Series:
    with sqlite3.connect(KP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["kp_index"]


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


def _build_severity_series(
    index: pd.DatetimeIndex, ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]]
) -> pd.Series:
    severity = pd.Series(0, index=index, dtype="Int8")
    grade_to_class = GRADE_TO_CLASS
    for start, end, grade in ranges:
        cls = grade_to_class.get(grade, 0)
        mask = (severity.index >= start) & (severity.index <= end)
        severity.loc[mask] = cls
    return severity


def _contiguous_positive_range(
    dst_series: pd.Series, start: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    before = dst_series[dst_series.index < start]
    if before.empty:
        return None
    if before.iloc[-1] < 0:
        return None
    values = before.to_numpy()
    idx = before.index
    end_idx = len(values) - 1
    start_idx = end_idx
    while start_idx >= 0 and values[start_idx] >= 0:
        start_idx -= 1
    start_idx += 1
    if start_idx >= len(values):
        return None
    span_start = idx[start_idx]
    span_end = start  # exclusive
    return span_start, span_end


def _ssc_ranges(
    dst_series: pd.Series, ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]]
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for start, _, grade in ranges:
        span = _contiguous_positive_range(dst_series, start)
        if span is not None:
            cls = GRADE_TO_CLASS.get(grade, 0)
            if cls > 0:
                spans.append((span[0], span[1], cls))
    return spans


def _main_phase_ranges(
    dst_series: pd.Series, ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]]
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for start, end, grade in ranges:
        storm_slice = dst_series[(dst_series.index >= start) & (dst_series.index <= end)]
        if storm_slice.empty:
            continue
        min_idx = storm_slice.idxmin()
        if pd.isna(min_idx):
            continue
        cls = GRADE_TO_CLASS.get(grade, 0)
        if cls > 0:
            spans.append((start, min_idx, cls))
    return spans


def _build_label_series(
    index: pd.DatetimeIndex,
    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    include_end: bool,
) -> pd.Series:
    series = pd.Series(0, index=index, dtype="Int8")
    for start, end, value in spans:
        if include_end:
            mask = (series.index >= start) & (series.index <= end)
        else:
            mask = (series.index >= start) & (series.index < end)
        series.loc[mask] = value
    return series


def _split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask].copy()


def build_severity_targets() -> None:
    dst = _load_dst()
    ap = _load_ap()
    kp = _load_kp()

    common_index = dst.index.intersection(ap.index).intersection(kp.index)
    dst = dst.loc[common_index]
    ap = ap.loc[common_index]
    kp = kp.loc[common_index]

    peaks = _find_peaks(ap)
    ranges = _deduplicate_ranges(_compute_ranges(dst, ap, kp, peaks))

    severity_series = _build_severity_series(dst.index, ranges)
    df = pd.DataFrame({"severity_label": severity_series})

    ssc_spans = _ssc_ranges(dst, ranges)
    main_spans = _main_phase_ranges(dst, ranges)
    ssc_series = _build_label_series(dst.index, ssc_spans, include_end=True)
    main_series = _build_label_series(dst.index, main_spans, include_end=True)

    ssc_df = pd.DataFrame({"ssc_label": ssc_series})
    main_df = pd.DataFrame({"main_phase_label": main_series})

    train_df = _split(df, TRAIN_START, TRAIN_END)
    val_df = _split(df, VAL_START, VAL_END)
    test_df = _split(df, TEST_START, TEST_END)

    train_ssc = _split(ssc_df, TRAIN_START, TRAIN_END)
    val_ssc = _split(ssc_df, VAL_START, VAL_END)
    test_ssc = _split(ssc_df, TEST_START, TEST_END)

    train_main = _split(main_df, TRAIN_START, TRAIN_END)
    val_main = _split(main_df, VAL_START, VAL_END)
    test_main = _split(main_df, TEST_START, TEST_END)

    with sqlite3.connect(OUTPUT_DB) as conn:
        train_df.to_sql(TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val_df.to_sql(VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test_df.to_sql(TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

        train_ssc.to_sql(SSC_TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val_ssc.to_sql(SSC_VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test_ssc.to_sql(SSC_TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

        train_main.to_sql(MAIN_TRAIN_TABLE, conn, if_exists="replace", index_label="timestamp")
        val_main.to_sql(MAIN_VAL_TABLE, conn, if_exists="replace", index_label="timestamp")
        test_main.to_sql(MAIN_TEST_TABLE, conn, if_exists="replace", index_label="timestamp")

    print(f"[OK] Storm severity labels written to {OUTPUT_DB}")
    print(f"     Train rows: {len(train_df):,}")
    print(f"     Validation rows: {len(val_df):,}")
    print(f"     Test rows: {len(test_df):,}")
    print("     SSC rows (train/val/test): "
          f"{len(train_ssc):,} / {len(val_ssc):,} / {len(test_ssc):,}")
    print("     Main phase rows (train/val/test): "
          f"{len(train_main):,} / {len(val_main):,} / {len(test_main):,}")


def main() -> None:
    build_severity_targets()


if __name__ == "__main__":
    main()
