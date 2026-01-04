from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing_pipeline.utils import read_timeseries_table

STAGE_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

IMF_DB = (
    PIPELINE_ROOT
    / "imf_solar_wind"
    / "6_engineered_features"
    / "imf_solar_wind_aver_comb_filt_imp_eng.db"
)
IMF_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "disturbed_labels.db"
TRAIN_TABLE = "disturbed_train"
VAL_TABLE = "disturbed_validation"
TEST_TABLE = "disturbed_test"

REQUIRED_CONDITIONS = 2

ENABLE_MIN_DURATION = True
MIN_DISTURBANCE_HOURS = 4

ENABLE_MERGE = False
SHORT_WINDOW_HOURS = 2
MERGE_GAP_HOURS = 2

TRAIN_START = pd.Timestamp("1999-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2016-12-31T23:59:59Z")
VAL_START = pd.Timestamp("2017-01-01T00:00:00Z")
VAL_END = pd.Timestamp("2020-12-31T23:59:59Z")
TEST_START = pd.Timestamp("2021-01-01T00:00:00Z")
TEST_END = pd.Timestamp("2025-11-30T23:59:59Z")


def build_disturbed_labels() -> pd.DataFrame:
    cols = [
        "bz_gse",
        "by_gse",
        "bt",
        "newell_dphi_dt",
        "speed",
        "dynamic_pressure",
    ]

    df = read_timeseries_table(
        IMF_TABLE,
        time_col="time_tag",
        value_cols=cols,
        db_path=IMF_DB,
    )

    if df.empty:
        raise RuntimeError("IMF/solar wind dataset is empty.")

    # -------------------------------------------------
    # Core IMF / solar wind conditions
    # -------------------------------------------------

    bz_south = df["bz_gse"] <= -2
    southward_flag = bz_south.rolling(4, min_periods=4).sum() >= 3

    # IMF clock angle (Y–Z plane)
    clock_angle = np.degrees(np.arctan2(df["by_gse"].abs(), df["bz_gse"]))
    southward_clock = clock_angle > 150.0
    clock_flag = southward_clock.rolling(3, min_periods=2).mean() >= 0.8

    bt_high = (df["bt"] >= 8.0).rolling(4, min_periods=4).sum() >= 2.5
    bt_jump = df["bt"].diff() > 2

    ey = df["speed"] * (-df["bz_gse"].clip(upper=0))
    ey_high = ey.rolling(3, min_periods=2).mean() > 1.5e3
    high_speed = ey.rolling(4, min_periods=3).mean() >= 2.5e3

    compression = df["dynamic_pressure"].pct_change() > 0.5

    strong_coupling = df["newell_dphi_dt"].rolling(4, min_periods=3).sum() > 1.5e4

    # -------------------------------------------------
    # Disturbed gate
    # -------------------------------------------------

    condition_count = (
        southward_flag.astype(int)
        + clock_flag.astype(int)
        + bt_high.astype(int)
        + bt_jump.astype(int)
        + ey_high.astype(int)
        + strong_coupling.astype(int)
        + high_speed.astype(int)
        + compression.astype(int)
    )

    disturbed_flag = (condition_count >= REQUIRED_CONDITIONS).fillna(False)

    # -------------------------------------------------
    # Duration / merging logic
    # -------------------------------------------------

    if (ENABLE_MERGE or ENABLE_MIN_DURATION) and disturbed_flag.any():
        run_id = disturbed_flag.ne(disturbed_flag.shift(fill_value=False)).cumsum()

        if ENABLE_MERGE:
            true_runs = disturbed_flag[disturbed_flag]
            run_ids = run_id[disturbed_flag]
            run_lengths = run_ids.value_counts().sort_index()
            run_starts = true_runs.groupby(run_ids).apply(lambda x: x.index[0])
            run_ends = true_runs.groupby(run_ids).apply(lambda x: x.index[-1])

            long_runs = run_lengths[run_lengths >= MIN_DISTURBANCE_HOURS]
            short_runs = run_lengths[run_lengths <= SHORT_WINDOW_HOURS]

            if not long_runs.empty and not short_runs.empty:
                max_gap = pd.Timedelta(hours=MERGE_GAP_HOURS)
                for short_id in short_runs.index:
                    for long_id in long_runs.index:
                        gap = min(
                            abs(run_starts[short_id] - run_ends[long_id]),
                            abs(run_starts[long_id] - run_ends[short_id]),
                        )
                        if gap <= max_gap:
                            disturbed_flag.loc[
                                min(run_starts[short_id], run_starts[long_id]) :
                                max(run_ends[short_id], run_ends[long_id])
                            ] = True
                            break

        if ENABLE_MIN_DURATION:
            run_id = disturbed_flag.ne(disturbed_flag.shift(fill_value=False)).cumsum()
            run_lengths = disturbed_flag.groupby(run_id).transform("sum")
            disturbed_flag = disturbed_flag & (run_lengths >= MIN_DISTURBANCE_HOURS)

    # -------------------------------------------------
    # Output tables
    # -------------------------------------------------

    labels = pd.DataFrame(
        {
            "time_tag": df.index,
            "disturbed_flag": disturbed_flag.astype(int),
        }
    )

    train = labels[(labels["time_tag"] >= TRAIN_START) & (labels["time_tag"] <= TRAIN_END)]
    val = labels[(labels["time_tag"] >= VAL_START) & (labels["time_tag"] <= VAL_END)]
    test = labels[(labels["time_tag"] >= TEST_START) & (labels["time_tag"] <= TEST_END)]

    with sqlite3.connect(OUTPUT_DB) as conn:
        train.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        val.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        test.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------

    print(f"[OK] Disturbed labels written to {OUTPUT_DB}")
    print(f"     Train / Val / Test rows: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"     Disturbed percentage: {labels['disturbed_flag'].mean()*100:.2f}%")
    print(f"     Southward IMF percentage: {southward_flag.mean()*100:.2f}%")
    print(f"     Southward clock percentage: {clock_flag.mean()*100:.2f}%")
    print(f"     High Bt percentage: {bt_high.mean()*100:.2f}%")
    print(f"     Bt jump percentage: {bt_jump.mean()*100:.2f}%")
    print(f"     High Ey percentage: {ey_high.mean()*100:.2f}%")
    print(f"     Strong coupling percentage: {strong_coupling.mean()*100:.2f}%")
    print(f"     High speed percentage: {high_speed.mean()*100:.2f}%")
    print(f"     Compression percentage: {compression.mean()*100:.2f}%")

    return labels


def main() -> None:
    build_disturbed_labels()


if __name__ == "__main__":
    main()
