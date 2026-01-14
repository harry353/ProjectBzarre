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
import sqlite3

from preprocessing_pipeline.utils import load_hourly_output

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

FILTERED_DB = (
    STAGE_DIR.parents[1]
    / "radio_flux"
    / "2_train_test_split"
    / "radio_flux_filt_split.db"
)
OUTPUT_DB = STAGE_DIR.parents[1] / "radio_flux" / "radio_flux_fin.db"
OUTPUT_TABLES = {
    "train": "radio_flux_train",
    "validation": "radio_flux_validation",
    "test": "radio_flux_test",
}

MEAN_WINDOW_H = 24
DELTA_WINDOW_H = 24
STD_WINDOW_H = 72
LOG_MEAN_WINDOW_H = 24
ANOMALY_WINDOW_H = 72
MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPS = 1e-6
SOLAR_ROTATION_HOURS = 27 * 24

# ---------------------------------------------------------------------
# Feature engineering (MINIMAL, HOURLY, CAUSAL)
# ---------------------------------------------------------------------
def _add_radio_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "adjusted_flux" not in working.columns:
        raise RuntimeError("Expected column 'adjusted_flux' missing.")

    # -----------------------------------------------------------------
    # Rename + ensure time index
    # -----------------------------------------------------------------
    working = working.rename(columns={"adjusted_flux": "f107"})

    if not isinstance(working.index, pd.DatetimeIndex):
        raise RuntimeError("Radio flux data must be time-indexed.")

    working = working.sort_index()
    if working.index.tz is None:
        working.index = working.index.tz_localize("UTC")
    else:
        working.index = working.index.tz_convert("UTC")

    # -----------------------------------------------------------------
    # Resample to hourly cadence (forward-fill, causal)
    # -----------------------------------------------------------------
    hourly_index = pd.date_range(
        working.index.min().floor("h"),
        working.index.max().ceil("h"),
        freq="1h",
        tz="UTC",
    )

    working = (
        working
        .reindex(hourly_index)
        .ffill()
    )

    # -----------------------------------------------------------------
    # Engineered features (ONLY 4)
    # -----------------------------------------------------------------
    working["log_f107"] = np.log(working["f107"].clip(lower=EPS))

    # Day-scale trend
    working["df107_24h"] = working["f107"] - working["f107"].shift(24)

    # Solar-rotation anomaly
    rolling_mean_27d = (
        working["f107"]
        .rolling(SOLAR_ROTATION_HOURS, min_periods=24)
        .mean()
    )
    working["f107_anomaly_27d"] = working["f107"] - rolling_mean_27d

    engineered = [
        "f107",
        "log_f107",
        "df107_24h",
        "f107_anomaly_27d",
    ]

    working[engineered] = working[engineered].fillna(0.0)

    if working[engineered].isna().any().any():
        raise RuntimeError("NaNs detected after radio flux feature engineering.")

    return working[engineered]


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def _min_periods(w: int) -> int:
    return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))


def _build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    f107 = pd.Series(df["f107"].to_numpy(dtype=float), index=df.index)
    log_f107 = pd.Series(df["log_f107"].to_numpy(dtype=float), index=df.index)

    w = MEAN_WINDOW_H
    window = f"{w}h"
    out[f"f107_mean_{w}h"] = (
        f107
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    w = DELTA_WINDOW_H
    out[f"f107_delta_{w}h"] = f107 - f107.shift(w)

    w = STD_WINDOW_H
    window = f"{w}h"
    out[f"f107_std_{w}h"] = (
        f107
        .rolling(window, min_periods=_min_periods(w))
        .std(ddof=0)
    )

    w = LOG_MEAN_WINDOW_H
    window = f"{w}h"
    out[f"log_f107_mean_{w}h"] = (
        log_f107
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    w = ANOMALY_WINDOW_H
    window = f"{w}h"
    out[f"f107_anomaly_{w}h"] = (
        f107
        - f107
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    out = out.dropna(subset=["f107", "log_f107", f"f107_mean_{MEAN_WINDOW_H}h", f"f107_delta_{DELTA_WINDOW_H}h", f"f107_std_{STD_WINDOW_H}h", f"log_f107_mean_{LOG_MEAN_WINDOW_H}h", f"f107_anomaly_{ANOMALY_WINDOW_H}h"])
    if out.empty:
        raise RuntimeError("No radio flux aggregate features produced.")

    return out


def engineer_radio_flux_features() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split, table in OUTPUT_TABLES.items():
        df = load_hourly_output(FILTERED_DB, table)
        if df.empty:
            raise RuntimeError("Filtered radio flux split not found; run split first.")

        features = _add_radio_flux_features(df)
        features = _build_aggregates(features)
        outputs[split] = features

        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[OK] Radio flux engineered+aggregate features written to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"Rows written ({split}): {len(features):,}")

    return outputs


def main() -> None:
    engineer_radio_flux_features()


if __name__ == "__main__":
    main()
