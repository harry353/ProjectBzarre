from __future__ import annotations

import os
import sys
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    raise RuntimeError("Project root not found (space_weather_api.py missing)")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STAGE_DIR = Path(__file__).resolve().parent

CATALOG_DB = STAGE_DIR.parents[1] / "cme" / "1_train_test_split" / "cme_catalog_split.db"
OUTPUT_DB = STAGE_DIR.parents[1] / "cme" / "cme_fin.db"

DEFAULT_WINDOWS = {
    "train": ("1999-01-01", "2016-12-31"),
    "validation": ("2017-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2025-11-30"),
}
SKIP_SPLITS = os.environ.get("PREPROC_SKIP_SPLITS", "").lower() in {"1", "true", "yes"}

DECAY_TAU_HOURS = 36.0
MIN_FRACTION_COVERAGE = 0.5
HOURS_SINCE_MIN_WINDOW_H = 6
V_MED_MAX_WINDOW_H = 12
INFLUENCE_MEAN_WINDOW_H = 12
SHOCK_MAX_WINDOW_H = 6


def _load_cme_catalog(table: str) -> pd.DataFrame:
    with sqlite3.connect(CATALOG_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {table}",
            conn,
            parse_dates=["time_tag", "timestamp"],
        )
    if "time_tag" in df.columns:
        df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
        df = df.dropna(subset=["time_tag"])
        return df.sort_values("time_tag")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        return df.rename(columns={"timestamp": "time_tag"}).sort_values("time_tag")
    raise RuntimeError("CME catalog split missing time_tag/timestamp column.")


def _parse_date(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _get_windows() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    env = os.environ
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split, (start_default, end_default) in DEFAULT_WINDOWS.items():
        start = env.get(f"PREPROC_SPLIT_{split.upper()}_START", start_default)
        end = env.get(f"PREPROC_SPLIT_{split.upper()}_END", end_default)
        windows[split] = (_parse_date(start), _parse_date(end))
    return windows


def _get_windows_from_splits() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split in ("train", "validation", "test"):
        table = f"cme_catalog_{split}"
        df = _load_cme_catalog(table)
        if df.empty:
            raise RuntimeError(f"CME split '{split}' is empty; cannot derive window.")
        time_col = "time_tag" if "time_tag" in df.columns else "timestamp"
        series = pd.to_datetime(df[time_col], utc=True, errors="coerce").dropna()
        if series.empty:
            raise RuntimeError(f"CME split '{split}' has no valid timestamps.")
        start = series.min().floor("h")
        end = series.max().ceil("h")
        windows[split] = (start, end)
    return windows


def _engineer_split(split: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    table = f"cme_catalog_{split}"
    cme = _load_cme_catalog(table)
    cme = cme.set_index("time_tag").sort_index()
    cme = cme.dropna(subset=["median_velocity"])
    cme = cme.loc[~cme.index.duplicated(keep="last")]

    cme["prev_cme_v_med"] = cme["median_velocity"].shift(1)
    cme["last_cme_speed_ratio"] = (
        cme["median_velocity"] / cme["prev_cme_v_med"].clip(lower=1.0)
    )
    cme["cme_overtaking_flag"] = (cme["last_cme_speed_ratio"] > 1.5).astype(int)

    cme["strength"] = cme["median_velocity"] * cme["angular_width"]
    cme["shock_proxy"] = (
        cme["median_velocity"]
        * (cme["velocity_variation"] / cme["median_velocity"].clip(lower=1.0))
    )

    pa = np.deg2rad(cme["position_angle"])
    earth_alignment = np.cos(pa).clip(-1.0, 1.0)
    cme["effective_width"] = cme["angular_width"] * earth_alignment.clip(0.0, 1.0)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    window_end = max(end, now)
    hourly_index = pd.date_range(start, window_end, freq="1h", tz="UTC")
    hourly = pd.DataFrame(index=hourly_index)
    hourly.index.name = "time_tag"

    event_index = cme.index.dropna().drop_duplicates().sort_values()
    if not event_index.is_monotonic_increasing:
        raise RuntimeError("CME event index not monotonic")

    event_series = pd.Series(event_index, index=event_index)
    last_event = event_series.reindex(hourly.index, method="ffill")

    hourly["hours_since_last_cme"] = (
        (hourly.index - last_event).dt.total_seconds() / 3600.0
    ).fillna(1e6)

    strength_series = (
        cme["strength"]
        .groupby(pd.Grouper(freq="1h"))
        .sum()
        .reindex(hourly.index, fill_value=0.0)
    )

    hourly["cme_strength_sum_24h"] = (
        strength_series.rolling(24, min_periods=1).sum().fillna(0.0)
    )

    def ff(series: pd.Series, fill=0.0) -> pd.Series:
        s = (
            series
            .dropna()
            .loc[~series.index.duplicated(keep="last")]
            .sort_index()
        )
        aligned = s.reindex(hourly.index)
        return aligned.ffill().fillna(fill)

    hourly["last_cme_v_med"] = ff(cme["median_velocity"])
    hourly["effective_width"] = ff(cme["effective_width"])
    hourly["cme_overtaking_flag"] = ff(cme["cme_overtaking_flag"], 0).astype(int)
    hourly["last_cme_shock_proxy"] = ff(cme["shock_proxy"])

    hourly["cme_influence_exp"] = np.exp(
        -hourly["hours_since_last_cme"] / DECAY_TAU_HOURS
    )

    hourly = hourly[
        [
            "hours_since_last_cme",
            "last_cme_v_med",
            "effective_width",
            "cme_strength_sum_24h",
            "cme_overtaking_flag",
            "cme_influence_exp",
            "last_cme_shock_proxy",
        ]
    ]

    return hourly


def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    out = df.copy()

    def _min_periods(w: int) -> int:
        return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))

    agg_cols: list[str] = []

    w = HOURS_SINCE_MIN_WINDOW_H
    window = f"{w}h"
    col = f"min_hours_since_last_cme_{w}h"
    out[col] = (
        df["hours_since_last_cme"]
        .rolling(window, min_periods=_min_periods(w))
        .min()
    )
    agg_cols.append(col)

    w = V_MED_MAX_WINDOW_H
    window = f"{w}h"
    col = f"max_last_cme_v_med_{w}h"
    out[col] = (
        df["last_cme_v_med"]
        .rolling(window, min_periods=_min_periods(w))
        .max()
    )
    agg_cols.append(col)

    w = INFLUENCE_MEAN_WINDOW_H
    window = f"{w}h"
    col = f"mean_cme_influence_exp_{w}h"
    out[col] = (
        df["cme_influence_exp"]
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )
    agg_cols.append(col)

    w = SHOCK_MAX_WINDOW_H
    window = f"{w}h"
    col = f"max_last_cme_shock_proxy_{w}h"
    out[col] = (
        df["last_cme_shock_proxy"]
        .rolling(window, min_periods=_min_periods(w))
        .max()
    )
    agg_cols.append(col)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=agg_cols)
    if out.empty:
        raise RuntimeError("No aggregated CME features produced.")

    return out


def engineer_cme_features() -> dict[str, pd.DataFrame]:
    windows = _get_windows_from_splits() if SKIP_SPLITS else _get_windows()
    outputs: dict[str, pd.DataFrame] = {}
    for split, (start, end) in windows.items():
        hourly = _engineer_split(split, start, end)
        outputs[split] = _build_agg(hourly)

    for split, features in outputs.items():
        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(f"cme_{split}", conn, if_exists="replace", index=False)

    print(f"[OK] CME engineered+aggregate features written to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"     {split}: {len(features):,} rows")

    return outputs


def main() -> None:
    engineer_cme_features()


if __name__ == "__main__":
    main()
