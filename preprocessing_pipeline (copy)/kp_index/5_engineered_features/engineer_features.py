from __future__ import annotations

import sys
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

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
IMPUTED_DB = (
    STAGE_DIR.parents[1]
    / "kp_index"
    / "4_imputation"
    / "kp_index_aver_filt_imp.db"
)
IMPUTED_TABLE = "imputed_data"

OUTPUT_DB = STAGE_DIR / "kp_index_aver_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

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
            "kp_delta_1h",
            "kp_delta_3h",
            "kp_delta_6h",
            "kp_ge5_flag",
            "kp_entered_storm",
        ]
    ]


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def engineer_kp_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed KP dataset not found; run imputation first.")

    features = _add_kp_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] KP engineered features saved to {OUTPUT_DB}")
    print(f"Rows written: {len(features):,}")

    return features


# ---------------------------------------------------------------------
def main() -> None:
    engineer_kp_features()


if __name__ == "__main__":
    main()

