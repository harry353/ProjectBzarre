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
KP_TO_AP = {
    0.00: 0, 0.33: 2, 0.67: 3, 1.00: 4, 1.33: 5, 1.67: 6, 2.00: 7,
    2.33: 9, 2.67: 12, 3.00: 15, 3.33: 18, 3.67: 22, 4.00: 27,
    4.33: 32, 4.67: 39, 5.00: 48, 5.33: 56, 5.67: 67, 6.00: 80,
    6.33: 94, 6.67: 111, 7.00: 132, 7.33: 154, 7.67: 179, 8.00: 207,
    8.33: 236, 8.67: 300, 9.00: 400,
}
KP_KEYS = np.array(sorted(KP_TO_AP.keys()), dtype=float)

# ---------------------------------------------------------------------
# Feature engineering (HOURLY, NO AGGREGATES)
# ---------------------------------------------------------------------
def _add_kp_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy().sort_index()

    if "kp_index" not in working.columns:
        raise RuntimeError("kp_index column missing from imputed dataset.")

    kp = working["kp_index"].astype(float)

    ap = kp.round(2).map(KP_TO_AP)
    missing = ap.isna()
    if missing.any():
        vals = kp.round(2)[missing].to_numpy()
        idx = np.abs(vals[:, None] - KP_KEYS[None, :]).argmin(axis=1)
        ap.loc[missing] = [KP_TO_AP[k] for k in KP_KEYS[idx]]
    working["ap"] = ap.astype(float)

    working["kp_regime"] = pd.cut(
        kp,
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    working["ap_3h_change"] = working["ap"].diff().fillna(0.0)

    working["ap_level_bucket"] = pd.cut(
        working["ap"],
        bins=[-np.inf, 10, 30, 80, 200, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    working = working.dropna()

    return working[
        [
            "ap",
            "kp_regime",
            "ap_3h_change",
            "ap_level_bucket",
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
