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
    working = df.copy()

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

    return working[
        [
            "kp_index",
            "ap",
            "kp_regime",
            "ap_3h_change",
            "ap_level_bucket",
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

