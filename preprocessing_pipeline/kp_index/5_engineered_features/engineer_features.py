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

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
IMPUTED_DB = STAGE_DIR.parents[1] / "kp_index" / "4_imputation" / "kp_index_aver_filt_imp.db"
IMPUTED_TABLE = "imputed_data"
OUTPUT_DB = STAGE_DIR / "kp_index_aver_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Kp ordinal-bin → Ap mapping (robust)
# ---------------------------------------------------------------------
KP_INDEX_TO_AP = [
    0, 2, 3, 4, 5, 6, 7, 9,
    12, 15, 18, 22, 27, 32,
    39, 48, 56, 67, 80, 94,
    111, 132, 154, 179, 207,
    236, 300, 400,
]


def kp_to_ap(kp: float) -> float:
    bin_index = int(np.round(kp * 3))
    bin_index = max(0, min(bin_index, len(KP_INDEX_TO_AP) - 1))
    return KP_INDEX_TO_AP[bin_index]


# ---------------------------------------------------------------------
# Feature engineering (NaN-free)
# ---------------------------------------------------------------------
def _add_kp_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy().sort_index()

    kp = working["kp_index"]

    # Raw Kp
    working["kp_index"] = kp

    # Linear Ap
    ap = kp.map(kp_to_ap)
    working["ap"] = ap

    # Regime (ordinal state)
    working["kp_regime"] = pd.cut(
        kp,
        bins=[-np.inf, 2, 4, 6, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Activity change (always defined, fill first diff with 0)
    working["ap_3h_change"] = ap.diff().fillna(0.0)

    # Nonlinear magnitude bucket (tree-friendly)
    working["ap_level_bucket"] = pd.cut(
        ap,
        bins=[-np.inf, 10, 30, 80, 200, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # ------------------------------------------------------------------
    # Tier-1 features
    # ------------------------------------------------------------------
    for lag in (1, 2, 3, 6, 12):
        working[f"kp_lag_{lag}"] = kp.shift(lag)

    working["kp_mean_6h"] = kp.rolling(window="6h", min_periods=1).mean()
    working["kp_mean_12h"] = kp.rolling(window="12h", min_periods=1).mean()
    working["kp_max_6h"] = kp.rolling(window="6h", min_periods=1).max()
    working["kp_max_12h"] = kp.rolling(window="12h", min_periods=1).max()
    working["kp_max_24h"] = kp.rolling(window="24h", min_periods=1).max()

    working["kp_delta_3h"] = kp - working["kp_lag_3"]
    working["kp_delta_6h"] = kp - working["kp_lag_6"]
    working["kp_accel"] = working["kp_delta_3h"] - working["kp_delta_6h"]

    def _consecutive_hours(series: pd.Series, threshold: float) -> pd.Series:
        mask = (series >= threshold).astype(int)
        groups = (mask == 0).cumsum()
        counts = mask.groupby(groups).cumsum()
        return counts.clip(upper=48)

    working["kp_hours_above_5"] = _consecutive_hours(kp, 5.0)
    working["kp_hours_above_6"] = _consecutive_hours(kp, 6.0)
    working["kp_hours_above_7"] = _consecutive_hours(kp, 7.0)

    regime = working["kp_regime"]
    regime_groups = (regime != regime.shift()).cumsum()
    working["kp_regime_duration_hours"] = regime.groupby(regime_groups).cumcount() + 1
    working["kp_time_since_last_regime_change"] = (
        working["kp_regime_duration_hours"] - 1
    )

    # ------------------------------------------------------------------
    # Tier-2 features
    # ------------------------------------------------------------------
    working["kp_dist_to_5"] = kp - 5.0
    working["kp_dist_to_6"] = kp - 6.0
    working["kp_dist_to_7"] = kp - 7.0

    working["ap_sum_24h"] = ap.rolling(window="24h", min_periods=1).sum()
    working["ap_max_24h"] = ap.rolling(window="24h", min_periods=1).max()
    working["ap_mean_12h"] = ap.rolling(window="12h", min_periods=1).mean()
    working["ap_energy_rolling"] = (ap.pow(2)).rolling(window="24h", min_periods=1).sum()

    working["kp_jump_2plus"] = (
        (kp - working["kp_lag_3"]) >= 2.0
    ).astype("Int8")
    working["kp_jump_3plus"] = (
        (kp - working["kp_lag_3"]) >= 3.0
    ).astype("Int8")
    working["kp_entered_storm"] = (
        (kp >= 5.0) & (working["kp_lag_1"] < 5.0)
    ).astype("Int8")

    # Fill NaNs introduced by lags/rolling
    numeric_cols = [
        col for col in working.columns if working[col].dtype.kind in "fcbiu"
    ]
    working[numeric_cols] = working[numeric_cols].fillna(0.0)

    return working[
        [
            "kp_index",
            "ap",
            "kp_regime",
            "ap_3h_change",
            "ap_level_bucket",
            "kp_lag_1",
            "kp_lag_2",
            "kp_lag_3",
            "kp_lag_6",
            "kp_lag_12",
            "kp_mean_6h",
            "kp_mean_12h",
            "kp_max_6h",
            "kp_max_12h",
            "kp_max_24h",
            "kp_delta_3h",
            "kp_delta_6h",
            "kp_accel",
            "kp_hours_above_5",
            "kp_hours_above_6",
            "kp_hours_above_7",
            "kp_regime_duration_hours",
            "kp_time_since_last_regime_change",
            "kp_dist_to_5",
            "kp_dist_to_6",
            "kp_dist_to_7",
            "ap_sum_24h",
            "ap_max_24h",
            "ap_mean_12h",
            "ap_energy_rolling",
            "kp_jump_2plus",
            "kp_jump_3plus",
            "kp_entered_storm",
        ]
    ]


# ---------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------
def engineer_kp_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed KP dataset not found.")

    features = _add_kp_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] KP engineered features saved to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_kp_features()


if __name__ == "__main__":
    main()
