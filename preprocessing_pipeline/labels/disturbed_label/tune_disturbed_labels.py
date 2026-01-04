from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sqlite3

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

PIPELINE_ROOT = PROJECT_ROOT / "preprocessing_pipeline"

IMF_DB = (
    PIPELINE_ROOT
    / "imf_solar_wind"
    / "6_engineered_features"
    / "imf_solar_wind_aver_comb_filt_imp_eng.db"
)
IMF_TABLE = "engineered_features"

SSC_DB = PIPELINE_ROOT / "features_targets" / "full_storm_label" / "full_storm_labels.db"
SSC_TABLES = [
    "storm_full_storm_train",
    "storm_full_storm_validation",
    "storm_full_storm_test",
]

LEAD_HOURS_LIST = [6, 8, 12]

EVAL_START: pd.Timestamp | None = None
EVAL_END: pd.Timestamp | None = None

TOP_N = 10

BZ_THRESHOLDS = [5.0, 6.0, 7.0]
NEWELL_THRESHOLDS = [2.0e4, 2.5e4, 3.0e4]
SPEED_THRESHOLDS = [500.0, 600.0, 700.0]
COMPRESSION_PCT_THRESHOLDS = [0.6, 0.8, 1.0]
REQUIRED_CONDITIONS = [1, 2, 3]

ENABLE_MERGE = False
SHORT_WINDOW_HOURS = 2
MERGE_GAP_HOURS = 2

ENABLE_MIN_DURATION = False
MIN_DISTURBANCE_HOURS = 6


@dataclass(frozen=True)
class ParamSet:
    bz_threshold: float
    newell_threshold: float
    speed_threshold: float
    compression_pct: float
    required_conditions: int


def _load_imf() -> pd.DataFrame:
    cols = ["bz_gse", "newell_dphi_dt", "speed", "dynamic_pressure"]
    df = read_timeseries_table(
        IMF_TABLE,
        time_col="time_tag",
        value_cols=cols,
        db_path=IMF_DB,
    )
    if df.empty:
        raise RuntimeError("IMF/solar wind dataset is empty.")
    return df


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
                    f"SELECT timestamp, storm_flag FROM {table}",
                    conn,
                    parse_dates=["timestamp"],
                )
                frames.append(frame)
    if not frames:
        raise RuntimeError("SSC labels not found.")
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


def _storm_onsets(full_storm: pd.DataFrame) -> list[pd.Timestamp]:
    active = full_storm["storm_flag"] == 1
    run_id = active.ne(active.shift(fill_value=False)).cumsum()
    onsets = []
    for run, flag in active.groupby(run_id):
        if flag.iloc[0]:
            onsets.append(full_storm.index[run_id == run][0])
    return onsets


def _apply_merge_min_duration(flag: pd.Series) -> pd.Series:
    if not (ENABLE_MERGE or ENABLE_MIN_DURATION) or not flag.any():
        return flag
    run_id = flag.ne(flag.shift(fill_value=False)).cumsum()
    true_runs = flag[flag]
    run_ids = run_id[flag]
    run_lengths = run_ids.value_counts().sort_index()
    run_starts = true_runs.groupby(run_ids).apply(lambda x: x.index[0])
    run_ends = true_runs.groupby(run_ids).apply(lambda x: x.index[-1])

    long_runs = run_lengths[run_lengths >= MIN_DISTURBANCE_HOURS]
    short_runs = run_lengths[run_lengths <= SHORT_WINDOW_HOURS]

    if ENABLE_MERGE and not long_runs.empty and not short_runs.empty:
        max_gap = pd.Timedelta(hours=MERGE_GAP_HOURS)
        for short_id in short_runs.index:
            short_start = run_starts[short_id]
            short_end = run_ends[short_id]
            for long_id in long_runs.index:
                long_start = run_starts[long_id]
                long_end = run_ends[long_id]
                if short_start > long_end:
                    gap = short_start - long_end
                elif long_start > short_end:
                    gap = long_start - short_end
                else:
                    gap = pd.Timedelta(0)
                if gap <= max_gap:
                    merge_start = min(long_start, short_start)
                    merge_end = max(long_end, short_end)
                    flag.loc[merge_start:merge_end] = True
                    break

    if ENABLE_MIN_DURATION:
        run_id = flag.ne(flag.shift(fill_value=False)).cumsum()
        run_lengths = flag.groupby(run_id).transform("sum")
        flag = flag & (run_lengths >= MIN_DISTURBANCE_HOURS)

    return flag


def _compute_disturbed(df: pd.DataFrame, params: ParamSet) -> pd.Series:
    bz_south = df["bz_gse"] <= -params.bz_threshold
    southward_flag = bz_south.rolling(2, min_periods=2).sum() >= 2
    strong_coupling = (
        df["newell_dphi_dt"].rolling(3, min_periods=3).sum() > params.newell_threshold
    )
    high_speed = (df["speed"] >= params.speed_threshold).rolling(2, min_periods=2).sum() >= 2
    compression = df["dynamic_pressure"].pct_change() > params.compression_pct

    condition_count = (
        southward_flag.astype(int)
        + strong_coupling.astype(int)
        + high_speed.astype(int)
        + compression.astype(int)
    )
    disturbed_flag = (condition_count >= params.required_conditions).fillna(False)
    return _apply_merge_min_duration(disturbed_flag)


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


def _evaluate(
    disturbed: pd.Series, storm_onsets: list[pd.Timestamp]
) -> tuple[float, float, dict[int, tuple[float, float]]]:
    per_lead = {}
    recalls = []
    precisions = []
    for h in LEAD_HOURS_LIST:
        r = stage1_recall(disturbed, storm_onsets, h)
        p = stage1_precision(disturbed, storm_onsets, h)
        per_lead[h] = (r, p)
        if not pd.isna(r):
            recalls.append(r)
        if not pd.isna(p):
            precisions.append(p)
    mean_recall = float(pd.Series(recalls).mean()) if recalls else float("nan")
    mean_precision = float(pd.Series(precisions).mean()) if precisions else float("nan")
    return mean_recall, mean_precision, per_lead


def main() -> None:
    imf = _load_imf()
    full_storm = _load_full_storm_labels()

    common_index = imf.index.intersection(full_storm.index)
    imf = imf.loc[common_index]
    full_storm = full_storm.loc[common_index]

    if EVAL_START is not None and EVAL_END is not None:
        imf = imf.loc[(imf.index >= EVAL_START) & (imf.index <= EVAL_END)]
        full_storm = full_storm.loc[(full_storm.index >= EVAL_START) & (full_storm.index <= EVAL_END)]

    storm_onsets = _storm_onsets(full_storm)

    param_grid = [
        ParamSet(bz, newell, speed, comp, req)
        for bz, newell, speed, comp, req in itertools.product(
            BZ_THRESHOLDS,
            NEWELL_THRESHOLDS,
            SPEED_THRESHOLDS,
            COMPRESSION_PCT_THRESHOLDS,
            REQUIRED_CONDITIONS,
        )
    ]

    results = []
    for params in param_grid:
        disturbed = _compute_disturbed(imf, params)
        mean_recall, mean_precision, per_lead = _evaluate(disturbed, storm_onsets)
        results.append((mean_recall, mean_precision, per_lead, params))

    results.sort(key=lambda x: (x[0], x[1]), reverse=True)

    print("Top parameter sets (sorted by mean recall, then mean precision):")
    for mean_recall, mean_precision, per_lead, params in results[:TOP_N]:
        detail = " ".join(
            f"{h}h r={per_lead[h][0]:.3f} p={per_lead[h][1]:.3f}"
            for h in LEAD_HOURS_LIST
        )
        print(
            f"  recall={mean_recall:.3f} precision={mean_precision:.3f} "
            f"bz={params.bz_threshold:.1f} "
            f"newell={params.newell_threshold:.1e} "
            f"speed={params.speed_threshold:.0f} "
            f"comp={params.compression_pct:.2f} "
            f"req={params.required_conditions} "
            f"| {detail}"
        )


if __name__ == "__main__":
    main()
