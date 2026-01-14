from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
from scipy.signal import find_peaks

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

DST_DB = PIPELINE_ROOT / "dst" / "1_averaging" / "dst_aver.db"
AP_DB = PIPELINE_ROOT / "kp_index" / "6_engineered_features" / "kp_index_aver_filt_imp_eng.db"
KP_DB = PIPELINE_ROOT / "kp_index" / "1_averaging" / "kp_index_aver.db"

OUTPUT_DB = STAGE_DIR / "storm_labels_daily_onset.db"
TRAIN_TABLE = "storm_daily_train"
VAL_TABLE = "storm_daily_validation"
TEST_TABLE = "storm_daily_test"

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
PEAK_PROMINENCE = 39.0
MAX_PRE_PEAK_OFFSET = pd.Timedelta(hours=24)
FORECAST_HORIZON = pd.Timedelta(hours=8)
CADENCE = pd.Timedelta(hours=8)

TRAIN_START = pd.Timestamp("1999-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2016-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2017-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2020-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2021-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-11-30T23:59:59Z")

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _ensure_utc(series: pd.Series) -> pd.Series:
    return (
        series.dt.tz_localize("UTC")
        if series.dt.tz is None
        else series.dt.tz_convert("UTC")
    )


def _load_series(db: Path, query: str, col: str) -> pd.Series:
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
    df["timestamp"] = _ensure_utc(df["timestamp"])
    return df.set_index("timestamp")[col]


def _find_peaks(ap: pd.Series) -> list[pd.Timestamp]:
    peaks, _ = find_peaks(ap.to_numpy(), prominence=PEAK_PROMINENCE)
    return [ap.index[i] for i in peaks]


# ---------------------------------------------------------------------
# Storm onset detection (MAIN PHASE ONSET)
# ---------------------------------------------------------------------
def _storm_onsets(dst: pd.Series, ap: pd.Series, kp: pd.Series) -> list[pd.Timestamp]:
    peaks = _find_peaks(ap)
    onsets: list[pd.Timestamp] = []

    for peak in peaks:
        window = dst[
            (dst.index <= peak) & (dst.index >= peak - MAX_PRE_PEAK_OFFSET)
        ]
        if window.empty:
            continue
        crossed = (window.shift(1) > 0) & (window <= 0)
        if not crossed.any():
            continue
        onset = crossed[crossed].index[0]

        dst_window = dst[(dst.index >= onset) & (dst.index <= peak)]
        kp_window = kp[(kp.index >= onset) & (kp.index <= peak)]

        if dst_window.min() > -50:
            continue
        if kp_window.max() < 5:
            continue

        onsets.append(onset)

    return sorted(set(onsets))


# ---------------------------------------------------------------------
# 8H LABELING: storm starts in NEXT 8 HOURS
# ---------------------------------------------------------------------
def _build_8h_labels(
    index: pd.DatetimeIndex,
    onsets: list[pd.Timestamp],
) -> pd.DataFrame:
    start = index.min().normalize()
    end = index.max().normalize() + pd.Timedelta(days=1)

    timestamps = pd.date_range(start, end, freq=CADENCE, tz="UTC")

    rows = []
    for t in timestamps:
        label = int(
            any((o > t) and (o <= t + FORECAST_HORIZON) for o in onsets)
        )
        rows.append(
            {
                "timestamp": t,
                "storm_starts_next_8h": label,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dst = _load_series(
        DST_DB,
        "SELECT time_tag AS timestamp, dst FROM hourly_data",
        "dst",
    )
    ap = _load_series(
        AP_DB,
        "SELECT time_tag AS timestamp, ap FROM engineered_features",
        "ap",
    )
    kp = _load_series(
        KP_DB,
        "SELECT time_tag AS timestamp, kp_index FROM hourly_data",
        "kp_index",
    )

    common = dst.index.intersection(ap.index).intersection(kp.index)
    dst, ap, kp = dst.loc[common], ap.loc[common], kp.loc[common]

    onsets = _storm_onsets(dst, ap, kp)
    labels = _build_8h_labels(common, onsets)

    train = labels[
        (labels["timestamp"] >= TRAIN_START)
        & (labels["timestamp"] <= TRAIN_END)
    ].copy()

    val = labels[
        (labels["timestamp"] >= VAL_START)
        & (labels["timestamp"] <= VAL_END)
    ].copy()

    test = labels[
        (labels["timestamp"] >= TEST_START)
        & (labels["timestamp"] <= TEST_END)
    ].copy()

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    print("[OK] 8h storm-onset labels written")
    print(f"Train / Val / Test rows: {len(train):,} / {len(val):,} / {len(test):,}")


if __name__ == "__main__":
    main()
