from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

STAGE_DIR = Path(__file__).resolve().parent
IMPUTED_DB = STAGE_DIR.parents[1] / "sunspot_number" / "4_imputation" / "sunspot_number_aver_filt_imp.db"
IMPUTED_TABLE = "imputed_data"
OUTPUT_DB = STAGE_DIR / "sunspot_number_aver_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

HOURS_IN_DAY = 24
ROLLING_81D = 81 * HOURS_IN_DAY
ROLLING_27D = 27 * HOURS_IN_DAY
ROLLING_365D = 365 * HOURS_IN_DAY
MAX_PERSISTENCE_DAYS = 180
CYCLE_MINIMA = [
    ("23", pd.Timestamp("1996-08-01T00:00:00Z")),
    ("24", pd.Timestamp("2008-12-01T00:00:00Z")),
    ("25", pd.Timestamp("2019-12-01T00:00:00Z")),
    ("26", pd.Timestamp("2031-01-01T00:00:00Z")),
]


def _assign_cycles(index: pd.DatetimeIndex) -> pd.DataFrame:
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    cycle_ids = pd.Series(index=index, dtype="object")
    cycle_start = pd.Series(index=index, dtype="datetime64[ns, UTC]")
    cycle_end = pd.Series(index=index, dtype="datetime64[ns, UTC]")

    for (cycle_id, start), (_, end) in zip(CYCLE_MINIMA[:-1], CYCLE_MINIMA[1:]):
        mask = (index >= start) & (index < end)
        cycle_ids.loc[mask] = cycle_id
        cycle_start.loc[mask] = start
        cycle_end.loc[mask] = end

    # Open-ended final known cycle
    last_id, last_start = CYCLE_MINIMA[-2]
    last_end = CYCLE_MINIMA[-1][1]

    mask_last = index >= last_start
    cycle_ids.loc[mask_last] = last_id
    cycle_start.loc[mask_last] = last_start
    cycle_end.loc[mask_last] = last_end

    return pd.DataFrame(
        {
            "cycle_id": cycle_ids,
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
        },
        index=index,
    )


def _rolling_slope(series: pd.Series, window_hours: int) -> pd.Series:
    x = np.arange(window_hours, dtype=float) / HOURS_IN_DAY
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y = values
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return series.rolling(window_hours, min_periods=window_hours).apply(_slope, raw=True)


def _cycle_expanding_quantile(
    values: pd.Series,
    cycle_ids: pd.Series,
    q: float = 0.6,
) -> pd.Series:
    return (
        values
        .groupby(cycle_ids)
        .expanding(min_periods=ROLLING_365D)
        .quantile(q)
        .reset_index(level=0, drop=True)
    )


def _persistence(values: pd.Series, cycle_ids: pd.Series, thresholds: pd.Series) -> pd.Series:
    mask = (values > thresholds).fillna(False)
    run = mask.astype(int)
    group_ids = (mask != mask.shift()).cumsum()
    run = run.groupby(group_ids).cumsum()
    run[~mask] = 0
    hours = run.clip(upper=MAX_PERSISTENCE_DAYS * HOURS_IN_DAY)
    days = (hours // HOURS_IN_DAY).clip(upper=MAX_PERSISTENCE_DAYS)
    days[~mask] = 0
    return days


STD_FLOOR = 1e-6


def _add_sunspot_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    target = working.get("sunspot_number")
    if target is None:
        raise RuntimeError("sunspot_number column missing from imputed dataset.")

    target = target.copy()
    cycle_meta = _assign_cycles(working.index)
    cycle_ids = cycle_meta["cycle_id"]

    working["ssn_raw"] = target
    working["ssn_log"] = np.log1p(target.clip(lower=0.0))
    working["ssn_mean_81d"] = target.rolling(
        ROLLING_81D, min_periods=ROLLING_81D // 2
    ).mean()
    working["ssn_slope_27d"] = _rolling_slope(target, ROLLING_27D)

    start = cycle_meta["cycle_start"]
    end = cycle_meta["cycle_end"]
    phase = ((working.index - start) / (end - start)).astype(float)
    working["ssn_cycle_phase"] = phase.clip(0.0, 1.0)

    working["ssn_lag_81d"] = target.shift(81 * HOURS_IN_DAY)

    mu_365 = target.rolling(
        ROLLING_365D, min_periods=ROLLING_365D
    ).mean()

    cycle_std = (
        target
        .groupby(cycle_ids)
        .expanding()
        .std()
        .reset_index(level=0, drop=True)
    )

    safe_cycle_std = cycle_std.clip(lower=STD_FLOOR)
    working["ssn_anomaly_cycle"] = (target - mu_365) / safe_cycle_std

    cycle_threshold = _cycle_expanding_quantile(target, cycle_ids, q=0.6)
    working["ssn_persistence"] = _persistence(
        target, cycle_ids, cycle_threshold
    )

    feature_columns = [
        "ssn_raw",
        "ssn_log",
        "ssn_mean_81d",
        "ssn_slope_27d",
        "ssn_cycle_phase",
        "ssn_lag_81d",
        "ssn_anomaly_cycle",
        "ssn_persistence",
    ]

    # HARD non-finite scrub
    working[feature_columns] = working[feature_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    working = working.dropna(subset=feature_columns)
    return working[feature_columns]


def engineer_sunspot_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed sunspot dataset not found; run imputation first.")
    features = _add_sunspot_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)
    print(f"[OK] Sunspot engineered features saved to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_sunspot_features()


if __name__ == "__main__":
    main()
