from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------
# Project root resolution
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

import numpy as np
import pandas as pd

from preprocessing_pipeline.utils import load_hourly_output

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "sunspot_number"
    / "5_train_test_split"
    / "sunspot_number_imputed_split.db"
)

OUTPUT_DB = STAGE_DIR.parents[1] / "sunspot_number" / "sunspot_number_fin.db"
OUTPUT_TABLES = {
    "train": "sunspot_train",
    "validation": "sunspot_validation",
    "test": "sunspot_test",
}

MEAN_STD_WINDOW_H = 81 * 24
TREND_WINDOW_H = 54 * 24
MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
HOURS_IN_DAY = 24
ROLLING_27D = 27 * HOURS_IN_DAY
ROLLING_81D = 81 * HOURS_IN_DAY
EPS = 1e-6

# ---------------------------------------------------------------------
# Feature engineering (MINIMAL, HOURLY)
# ---------------------------------------------------------------------
def _add_sunspot_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy().sort_index()

    if "sunspot_number" not in working.columns:
        raise RuntimeError("sunspot_number column missing from imputed dataset.")

    ssn = working["sunspot_number"].astype(float)

    # --------------------------------------------------------------
    # Core level
    # --------------------------------------------------------------
    working["ssn"] = ssn

    # --------------------------------------------------------------
    # Log-compressed activity
    # --------------------------------------------------------------
    working["log_ssn"] = np.log1p(ssn.clip(lower=0.0))

    # --------------------------------------------------------------
    # Long-term trend (solar cycle direction)
    # --------------------------------------------------------------
    lag_27d = ssn.shift(ROLLING_27D)
    working["ssn_slope_27d"] = (ssn - lag_27d) / 27.0

    # --------------------------------------------------------------
    # Background-relative activity
    # --------------------------------------------------------------
    mean_81d = ssn.rolling(ROLLING_81D, min_periods=1).mean()
    working["ssn_anomaly_81d"] = ssn - mean_81d

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------
    feature_cols = [
        "ssn",
        "log_ssn",
        "ssn_slope_27d",
        "ssn_anomaly_81d",
    ]

    working[feature_cols] = working[feature_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    return working[feature_cols]


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def _min_periods(w: int) -> int:
    return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))


def _add_sunspot_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ssn_series = pd.Series(df["ssn"].to_numpy(dtype=float), index=df.index)

    w = MEAN_STD_WINDOW_H
    window = f"{w}h"
    out[f"ssn_mean_{w}h"] = (
        ssn_series
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )
    out[f"ssn_std_{w}h"] = (
        ssn_series
        .rolling(window, min_periods=_min_periods(w))
        .std()
        .fillna(0.0)
    )

    w = TREND_WINDOW_H
    lagged = ssn_series.shift(w)
    out[f"ssn_trend_{w}h"] = (ssn_series - lagged) / (w / 24.0)

    mean_ref = out[f"ssn_mean_{MEAN_STD_WINDOW_H}h"]
    above = (ssn_series > mean_ref).astype(float)
    out[f"ssn_anomaly_frac_{MEAN_STD_WINDOW_H}h"] = (
        above
        .rolling(window, min_periods=_min_periods(MEAN_STD_WINDOW_H))
        .mean()
    )

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["ssn", f"ssn_mean_{MEAN_STD_WINDOW_H}h", f"ssn_std_{MEAN_STD_WINDOW_H}h", f"ssn_trend_{TREND_WINDOW_H}h", f"ssn_anomaly_frac_{MEAN_STD_WINDOW_H}h"])
    if out.empty:
        raise RuntimeError("No sunspot aggregate features produced.")

    return out


def engineer_sunspot_features() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split, table in OUTPUT_TABLES.items():
        df = load_hourly_output(SPLITS_DB, table)
        if df.empty:
            raise RuntimeError("Imputed sunspot split not found; run split first.")

        features = _add_sunspot_features(df)
        features = _add_sunspot_agg_features(features)
        outputs[split] = features

        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[OK] Sunspot engineered+aggregate features written to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"Rows written ({split}): {len(features):,}")

    return outputs


def main() -> None:
    engineer_sunspot_features()


if __name__ == "__main__":
    main()
