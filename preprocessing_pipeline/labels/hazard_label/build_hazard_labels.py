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
KP_DB = PROJECT_ROOT / "preprocessing_pipeline" / "kp_index" / "1_averaging" / "kp_index_aver.db"

OUTPUT_DB = STAGE_DIR / "storm_onset_hazards.db"
TRAIN_TABLE = "storm_onset_train"
VAL_TABLE = "storm_onset_validation"
TEST_TABLE = "storm_onset_test"

PEAK_PROMINENCE = 39.0
STORM_OVERLAP_ALLOWANCE = pd.Timedelta(hours=4)
FORECAST_HORIZONS_H = range(1, 9)

TRAIN_START = pd.Timestamp("1999-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2016-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2017-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2020-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2021-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-11-30T23:59:59Z")

KP_TO_AP = {
    0.00: 0, 0.33: 2, 0.67: 3, 1.00: 4, 1.33: 5, 1.67: 6, 2.00: 7,
    2.33: 9, 2.67: 12, 3.00: 15, 3.33: 18, 3.67: 22, 4.00: 27,
    4.33: 32, 4.67: 39, 5.00: 48, 5.33: 56, 5.67: 67, 6.00: 80,
    6.33: 94, 6.67: 111, 7.00: 132, 7.33: 154, 7.67: 179, 8.00: 207,
    8.33: 236, 8.67: 300, 9.00: 400,
}
KP_KEYS = np.array(sorted(KP_TO_AP.keys()), dtype=float)


def _ensure_utc(series: pd.Series) -> pd.Series:
    return series.dt.tz_localize("UTC") if series.dt.tz is None else series.dt.tz_convert("UTC")


def _load_dst() -> pd.Series:
    with sqlite3.connect(DST_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, dst FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["dst"].sort_index()


def _load_kp() -> pd.Series:
    with sqlite3.connect(KP_DB) as conn:
        df = pd.read_sql_query(
            "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
            conn,
            parse_dates=["timestamp"],
        )
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")["kp_index"].sort_index()


def _kp_to_ap(kp: pd.Series) -> pd.Series:
    kp_vals = kp.astype(float).round(2)
    mapped = kp_vals.map(KP_TO_AP)
    missing = mapped.isna()
    if missing.any():
        vals = kp_vals[missing].to_numpy()
        idx = np.abs(vals[:, None] - KP_KEYS[None, :]).argmin(axis=1)
        mapped.loc[missing] = [KP_TO_AP[k] for k in KP_KEYS[idx]]
    return mapped.astype(float)


def _find_peaks(ap: pd.Series) -> list[pd.Timestamp]:
    values = ap.to_numpy()
    peaks, _ = find_peaks(values, prominence=PEAK_PROMINENCE)
    return [ap.index[i] for i in peaks]


def _deduplicate_onsets(onsets: list[pd.Timestamp]) -> list[pd.Timestamp]:
    out = []
    for t in sorted(onsets):
        if not out or t - out[-1] >= STORM_OVERLAP_ALLOWANCE:
            out.append(t)
    return out


def _infer_onsets(dst: pd.Series, ap: pd.Series) -> list[pd.Timestamp]:
    peaks = _find_peaks(ap)
    prev = dst.shift(1)
    crossings = dst.index[(dst <= 0) & (prev > 0)]
    onsets = []
    for peak in peaks:
        prior = crossings[crossings <= peak]
        if len(prior) == 0:
            continue
        onsets.append(prior[-1])
    return _deduplicate_onsets(onsets)


def _build_interval_labels(
    index: pd.DatetimeIndex,
    onsets: list[pd.Timestamp],
) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    for h in FORECAST_HORIZONS_H:
        df[f"h_{h}"] = 0

    for onset in onsets:
        dt_hours = (onset - index).total_seconds() / 3600.0
        for h in FORECAST_HORIZONS_H:
            mask = (dt_hours > (h - 1)) & (dt_hours <= h)
            df.loc[mask, f"h_{h}"] = 1

    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def _split(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()


def _count_storms(df: pd.DataFrame) -> int:
    times = []
    for h in FORECAST_HORIZONS_H:
        col = f"h_{h}"
        mask = df[col] == 1
        if mask.any():
            times.append(df.loc[mask, "timestamp"] + pd.Timedelta(hours=h))
    if not times:
        return 0
    return len(pd.concat(times).drop_duplicates())


def build_hazard_labels() -> None:
    dst = _load_dst()
    kp = _load_kp()
    ap = _kp_to_ap(kp)

    idx = dst.index.intersection(ap.index)
    dst = dst.loc[idx]
    ap = ap.loc[idx]

    onsets = _infer_onsets(dst, ap)
    labels = _build_interval_labels(dst.index, onsets)

    train = _split(labels, TRAIN_START, TRAIN_END)
    val = _split(labels, VAL_START, VAL_END)
    test = _split(labels, TEST_START, TEST_END)

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    print("[OK] Interval hazard labels written to", OUTPUT_DB)
    print(f"     Train / Val / Test rows: {len(train):,} / {len(val):,} / {len(test):,}")
    print(
        f"     Train / Val / Test storms: "
        f"{_count_storms(train)} / {_count_storms(val)} / {_count_storms(test)}"
    )


def main() -> None:
    build_hazard_labels()


if __name__ == "__main__":
    main()
