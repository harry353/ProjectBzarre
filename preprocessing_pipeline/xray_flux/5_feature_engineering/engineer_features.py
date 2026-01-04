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

IMPUTED_DB = (
    STAGE_DIR.parents[1]
    / "xray_flux"
    / "4_imputation"
    / "xray_flux_aver_filt_imp.db"
)
IMPUTED_TABLE = "imputed_data"

OUTPUT_DB = STAGE_DIR / "xray_flux_aver_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPS = 1e-12
FLUX_A = "irradiance_xrsa"
FLUX_B = "irradiance_xrsb"

# ---------------------------------------------------------------------
# Feature engineering (MINIMAL, NO AGGREGATES)
# ---------------------------------------------------------------------
def _add_xrs_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy().sort_index()

    for col in ("irradiance_xrsa", "irradiance_xrsb"):
        if col not in working.columns:
            raise RuntimeError(f"Missing required column: {col}")

    a = working["irradiance_xrsa"].astype(float)
    b = working["irradiance_xrsb"].astype(float)

    # --------------------------------------------------------------
    # Core log fluxes
    # --------------------------------------------------------------
    working["log_xrsb"] = np.log10(b + EPS)
    working["log_xrsa"] = np.log10(a + EPS)

    # --------------------------------------------------------------
    # Spectral relationships
    # --------------------------------------------------------------
    working["xrs_hardness"] = working["log_xrsb"] - working["log_xrsa"]
    working["xrsa_to_xrsb_ratio_log"] = working["log_xrsa"] - working["log_xrsb"]

    # --------------------------------------------------------------
    # Temporal dynamics (impulsiveness)
    # --------------------------------------------------------------
    working["dlog_xrsb_1h"] = working["log_xrsb"].diff()
    working["dlog_xrsa_1h"] = working["log_xrsa"].diff()

    # --------------------------------------------------------------
    # Short memory (causal)
    # --------------------------------------------------------------
    working["log_xrsb_lag_1h"] = working["log_xrsb"].shift(1)
    working["log_xrsa_lag_1h"] = working["log_xrsa"].shift(1)

    engineered = [
        # XRS-B
        "log_xrsb",
        "dlog_xrsb_1h",
        "log_xrsb_lag_1h",

        # XRS-A
        "log_xrsa",
        "dlog_xrsa_1h",
        "log_xrsa_lag_1h",

        # Cross-channel
        "xrs_hardness",
        "xrsa_to_xrsb_ratio_log",
    ]

    # --------------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------------
    working[engineered] = working[engineered].fillna(0.0)

    if working[engineered].isna().any().any():
        raise RuntimeError("NaNs detected after X-ray feature engineering.")

    return working[engineered]


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def engineer_xrs_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed X-ray dataset not found.")

    features = _add_xrs_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] X-ray engineered features written to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_xrs_features()


if __name__ == "__main__":
    main()
