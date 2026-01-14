from __future__ import annotations

import sys
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

from preprocessing_pipeline.utils import load_hourly_output, write_sqlite_table

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent

IMPUTED_DB = (
    STAGE_DIR.parents[1]
    / "sunspot_number"
    / "4_imputation"
    / "sunspot_number_aver_filt_imp.db"
)
IMPUTED_TABLE = "imputed_data"

OUTPUT_DB = STAGE_DIR / "sunspot_number_aver_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

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
def engineer_sunspot_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed sunspot dataset not found; run imputation first.")

    features = _add_sunspot_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] Sunspot engineered features written to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_sunspot_features()


if __name__ == "__main__":
    main()
