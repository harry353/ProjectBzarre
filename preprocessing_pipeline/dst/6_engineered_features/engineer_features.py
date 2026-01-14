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
    / "dst"
    / "5_train_test_split"
    / "dst_imputed_split.db"
)

OUTPUT_DB = STAGE_DIR.parents[1] / "dst" / "dst_fin.db"
OUTPUT_TABLES = {
    "train": "dst_train",
    "validation": "dst_validation",
    "test": "dst_test",
}

WINDOW_H = 6
MIN_FRACTION_COVERAGE = 0.5

# ---------------------------------------------------------------------
# Feature engineering (HOURLY, NO AGGREGATES)
# ---------------------------------------------------------------------
def _add_dst_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    if "dst" not in working.columns:
        raise RuntimeError("DST column missing from imputed dataset.")

    dst = working["dst"].astype(float)

    working["dst"] = dst
    working["dst_mean_6h"] = working["dst"].rolling(window="6h", min_periods=1).mean()
    working["dst_std_6h"] = (
        working["dst"]
        .rolling(window="6h", min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
    )
    working["dst_derivative"] = (working["dst"] - working["dst"].shift(1)).fillna(0.0)

    return working[["dst", "dst_mean_6h", "dst_std_6h", "dst_derivative"]]

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


def _add_dst_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    min_periods = max(1, int(np.ceil(WINDOW_H * MIN_FRACTION_COVERAGE)))
    window = f"{WINDOW_H}h"

    out[f"dst_min_{WINDOW_H}h"] = (
        df["dst"]
        .rolling(window, min_periods=min_periods)
        .min()
    )

    out[f"dst_mean_{WINDOW_H}h"] = (
        df["dst"]
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out[f"dst_delta_{WINDOW_H}h"] = (
        df["dst"] - df["dst"].shift(WINDOW_H)
    )

    out[f"dst_slope_{WINDOW_H}h"] = (
        df["dst"]
        .rolling(window, min_periods=min_periods)
        .apply(_linear_slope, raw=False)
    )

    out[f"dst_neg_frac_{WINDOW_H}h"] = (
        (df["dst"] < 0)
        .rolling(window, min_periods=min_periods)
        .mean()
    )

    out = out.dropna()
    if out.empty:
        raise RuntimeError("No DST aggregate features produced.")

    return out


def engineer_dst_features() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split, table in OUTPUT_TABLES.items():
        df = load_hourly_output(SPLITS_DB, table)
        if df.empty:
            raise RuntimeError("Imputed DST split not found; run split first.")
        features = _add_dst_features(df)
#        features = _add_dst_agg_features(features)
        outputs[split] = features

        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[OK] DST engineered+aggregate features saved to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"Rows written ({split}): {len(features):,}")

    return outputs

# ---------------------------------------------------------------------
def main() -> None:
    engineer_dst_features()

if __name__ == "__main__":
    main()
