from __future__ import annotations

import sys
from pathlib import Path
import sqlite3

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

from preprocessing_pipeline.utils import load_hourly_output

STAGE_DIR = Path(__file__).resolve().parent
FEATURES_DB = (
    STAGE_DIR.parents[1]
    / "flares"
    / "3_engineered_features"
    / "flares_comb_filt_eng.db"
)
FEATURES_TABLE = "engineered_features"
OUTPUT_DB = STAGE_DIR / "flare_agg_eng.db"
OUTPUT_TABLE = "features_agg"

MIN_ROWS_PER_DAY = 12

def _load_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("Flare engineered dataset is empty; run feature engineering first.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected flare features indexed by timestamp.")
    df = df.sort_index()
    df.index = df.index.tz_convert("UTC")
    return df


def _build_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df.index.normalize()

    daily_rows = []
    for date, group in df.groupby("date"):
        if len(group) < MIN_ROWS_PER_DAY:
            continue
        group = group.sort_index()
        latest = group.iloc[-1]

        row = {
            "date": date,
            "flare_active_flag": int(group["flare_active_flag"].max()),
            "hours_since_last_flare": float(latest["hours_since_last_flare"]),
            "flare_count_last_24h": float(latest["flare_count_last_24h"]),
            "flare_count_last_72h": float(latest["flare_count_last_72h"]),
            "last_flare_peak_flux": float(latest["last_flare_peak_flux"]),
            "last_flare_integrated_flux": float(latest["last_flare_integrated_flux"]),
            "flare_class_ord": float(latest["flare_class_ord"]),
            "flare_major_flag": int(latest["flare_major_flag"]),
            "flare_extreme_flag": int(latest["flare_extreme_flag"]),
            "flare_influence_exp": float(latest["flare_influence_exp"]),
        }
        daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("No daily flare rows produced; check source coverage.")
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def create_daily_flare_features() -> pd.DataFrame:
    df = _load_features()
    daily = _build_daily_frame(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        daily.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] Daily flare aggregates saved to {OUTPUT_DB}")
    return daily


def main() -> None:
    create_daily_flare_features()


if __name__ == "__main__":
    main()
