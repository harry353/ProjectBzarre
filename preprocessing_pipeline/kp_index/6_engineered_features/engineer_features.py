from __future__ import annotations

import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root
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

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "kp_index"
    / "5_train_test_split"
    / "kp_imputed_split.db"
)

OUTPUT_DB = STAGE_DIR.parents[1] / "kp_index" / "kp_fin.db"
OUTPUT_TABLES = {
    "train": "kp_train",
    "validation": "kp_validation",
    "test": "kp_test",
}

WINDOW_H = 6
MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Feature engineering (HOURLY, NO AGGREGATES)
# ---------------------------------------------------------------------
def _add_kp_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy().sort_index()

    if "kp_index" not in working.columns:
        raise RuntimeError("kp_index column missing from imputed dataset.")

    kp = working["kp_index"].astype(float)

    # --------------------------------------------------------------
    # Core KP features (6 total)
    # --------------------------------------------------------------
    working["kp"] = kp

    working["kp_delta_1h"] = kp.diff(1)
    working["kp_delta_3h"] = kp.diff(3)
    working["kp_delta_6h"] = kp.diff(6)

    working["kp_ge5_flag"] = (kp >= 5.0).astype(int)

    working["kp_entered_storm"] = (
        (kp >= 5.0) & (kp.shift(1) < 5.0)
    ).astype(int)

    # --------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------
    working = working.dropna()

    return working[
        [
            "kp",
            # "kp_delta_1h",
            # "kp_delta_3h",
            # "kp_delta_6h",
            "kp_ge5_flag",
            "kp_entered_storm",
        ]
    ]


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def _linear_slope(series: pd.Series) -> float:
    y = series.to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan

    x = np.arange(len(y), dtype=float)[mask]
    y = y[mask]

    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0

    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _add_kp_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    min_periods = max(1, int(np.ceil(WINDOW_H * MIN_FRACTION_COVERAGE)))
    window = f"{WINDOW_H}h"

    out[f"kp_max_{WINDOW_H}h"] = (
        df["kp"]
        .rolling(window, min_periods=min_periods)
        .max()
    )

    out[f"kp_mean_{WINDOW_H}h"] = (
        df["kp"]
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out[f"kp_delta_{WINDOW_H}h"] = (
        df["kp"] - df["kp"].shift(WINDOW_H)
    )

    out[f"kp_ge5_frac_{WINDOW_H}h"] = (
        (df["kp"] >= 5.0)
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out[f"kp_slope_{WINDOW_H}h"] = (
        df["kp"]
        .rolling(window, min_periods=min_periods)
        .apply(_linear_slope, raw=False)
    )

    out = out.dropna()
    if out.empty:
        raise RuntimeError("No KP aggregate features produced.")

    return out


def engineer_kp_features() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split, table in OUTPUT_TABLES.items():
        df = load_hourly_output(SPLITS_DB, table)
        if df.empty:
            raise RuntimeError("Imputed KP split not found; run split first.")

        features = _add_kp_features(df)
#        features = _add_kp_agg_features(features)
        outputs[split] = features

        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[OK] KP engineered+aggregate features saved to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"Rows written ({split}): {len(features):,}")

    return outputs


# ---------------------------------------------------------------------
def main() -> None:
    engineer_kp_features()


if __name__ == "__main__":
    main()
