from __future__ import annotations

import sys
from pathlib import Path
import sqlite3
from typing import Dict, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
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
    / "xray_flux"
    / "5_feature_engineering"
    / "xray_flux_aver_filt_imp_eng.db"
)
FEATURES_TABLE = "engineered_features"

OUTPUT_DB = STAGE_DIR / "xray_flux_agg_eng.db"
OUTPUT_TABLE = "features_agg"

# ---------------------------------------------------------------------
# Aggregation parameters (AUTHORITATIVE)
# ---------------------------------------------------------------------
MIN_FRACTION_COVERAGE = 0.5

MEAN_WINDOW_H = 6
MAX_WINDOW_H = 24
SLOPE_WINDOW_H = 6
PEAK_BG_WINDOW_H = 24
RAPID_RISE_WINDOW_H = 24

# ---------------------------------------------------------------------
# Load hourly features
# ---------------------------------------------------------------------
def _load_hourly_features() -> pd.DataFrame:
    df = load_hourly_output(FEATURES_DB, FEATURES_TABLE)
    if df.empty:
        raise RuntimeError("X-ray engineered dataset is empty.")

    required = [
        "log_xrsb",
        "log_xrsa",
        "dlog_xrsb_1h",
        "dlog_xrsa_1h",
    ]

    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Required column '{col}' missing.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected X-ray features indexed by timestamp.")

    df = df.sort_index()
    df.index = df.index.tz_convert("UTC")

    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _min_periods(w: int) -> int:
    return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))


def _rolling_slope(series: pd.Series, w: int) -> pd.Series:
    def _calc(y: pd.Series) -> float:
        values = y.to_numpy(dtype=float)
        if np.isnan(values).any():
            return np.nan
        if len(values) < 2:
            return np.nan
        x = np.arange(len(values), dtype=float)
        xm = x.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0.0:
            return np.nan
        ym = values.mean()
        return float(((x - xm) * (values - ym)).sum() / denom)

    return series.rolling(f"{w}h", min_periods=_min_periods(w)).apply(_calc, raw=False)


# ---------------------------------------------------------------------
# Build aggregates (PAST-ONLY, HOURLY CADENCE)
# ---------------------------------------------------------------------
def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --------------------------------------------------------------
    # XRS-B aggregates (primary flare driver)
    # --------------------------------------------------------------
    w = MEAN_WINDOW_H
    out[f"log_xrsb_mean_{w}h"] = df["log_xrsb"].rolling(
        f"{w}h", min_periods=_min_periods(w)
    ).mean()

    w = MAX_WINDOW_H
    out[f"log_xrsb_max_{w}h"] = df["log_xrsb"].rolling(
        f"{w}h", min_periods=_min_periods(w)
    ).max()

    w = SLOPE_WINDOW_H
    out[f"log_xrsb_slope_{w}h"] = _rolling_slope(df["log_xrsb"], w)

    w = PEAK_BG_WINDOW_H
    bg = df["log_xrsb"].rolling(f"{w}h", min_periods=_min_periods(w)).min()
    out[f"log_xrsb_peak_to_bg_{w}h"] = df["log_xrsb"] - bg

    # --------------------------------------------------------------
    # XRS-A aggregates (context)
    # --------------------------------------------------------------
    w = MEAN_WINDOW_H
    out[f"log_xrsa_mean_{w}h"] = df["log_xrsa"].rolling(
        f"{w}h", min_periods=_min_periods(w)
    ).mean()

    w = MAX_WINDOW_H
    out[f"log_xrsa_max_{w}h"] = df["log_xrsa"].rolling(
        f"{w}h", min_periods=_min_periods(w)
    ).max()

    w = SLOPE_WINDOW_H
    out[f"log_xrsa_slope_{w}h"] = _rolling_slope(df["log_xrsa"], w)

    w = PEAK_BG_WINDOW_H
    bg = df["log_xrsa"].rolling(f"{w}h", min_periods=_min_periods(w)).min()
    out[f"log_xrsa_peak_to_bg_{w}h"] = df["log_xrsa"] - bg

    # --------------------------------------------------------------
    # Event recency (shared)
    # --------------------------------------------------------------
    w = RAPID_RISE_WINDOW_H
    out[f"hrs_since_rapid_rise_xrsb_{w}h"] = (
        (df["dlog_xrsb_1h"] > 0.3)
        .astype(int)
        .rolling(f"{w}h", min_periods=1)
        .apply(lambda x: np.argmax(x[::-1]) if x.any() else np.nan)
    )

    out[f"hrs_since_rapid_rise_xrsa_{w}h"] = (
        (df["dlog_xrsa_1h"] > 0.3)
        .astype(int)
        .rolling(f"{w}h", min_periods=1)
        .apply(lambda x: np.argmax(x[::-1]) if x.any() else np.nan)
    )

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------
    out = out.dropna()

    if out.empty:
        raise RuntimeError("No X-ray aggregate features produced.")

    return out.reset_index().rename(columns={"index": "timestamp"})


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def create_xray_agg_features() -> pd.DataFrame:
    df = _load_hourly_features()
    features = _build_agg(df)

    with sqlite3.connect(OUTPUT_DB) as conn:
        features.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] X-ray aggregate features written to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


def main() -> None:
    create_xray_agg_features()


if __name__ == "__main__":
    main()
