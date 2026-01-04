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
    / "imf_solar_wind"
    / "5_imputation"
    / "imf_solar_wind_aver_comb_filt_imp.db"
)
IMPUTED_TABLE = "imputed_data"
OUTPUT_DB = STAGE_DIR / "imf_solar_wind_aver_comb_filt_imp_eng.db"
OUTPUT_TABLE = "engineered_features"

# ---------------------------------------------------------------------
# Feature engineering (all NaN-safe, append-only)
# ---------------------------------------------------------------------
ESSENTIAL_COLUMNS = ["bx_gse", "by_gse", "bz_gse", "bt", "speed", "density"]


def _add_sw_imf_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    # ------------------------------------------------------------------
    # Required base columns (remain as-is)
    # ------------------------------------------------------------------
    required = ["bx_gse", "by_gse", "bz_gse", "bt", "speed", "density"]
    for col in required:
        if col not in working.columns:
            raise RuntimeError(f"Required column '{col}' missing from imputed dataset.")

    bx = working["bx_gse"].astype(float)
    by = working["by_gse"].astype(float)
    bz = working["bz_gse"].astype(float)
    bt = working["bt"].astype(float)
    v  = working["speed"].astype(float)
    n  = working["density"].astype(float)

    # ------------------------------------------------------------------
    # 1. Clock angle (IMF orientation)
    # ------------------------------------------------------------------
    clock = np.arctan2(by, bz)
    working["clock_angle"] = clock.fillna(0.0)

    # ------------------------------------------------------------------
    # 2. Newell coupling function
    # ------------------------------------------------------------------
    sin_half = np.sin(clock / 2.0)
    working["newell_dphi_dt"] = (
        (v ** (4.0 / 3.0))
        * (bt ** (2.0 / 3.0))
        * (np.abs(sin_half) ** (8.0 / 3.0))
    ).fillna(0.0)

    # ------------------------------------------------------------------
    # 3. Dayside reconnection electric field proxy
    # ------------------------------------------------------------------
    working["ey"] = (-v * bz).fillna(0.0)

    # ------------------------------------------------------------------
    # 4. Dynamic pressure (nPa)
    # ------------------------------------------------------------------
    mp = 1.6726219e-27
    working["dynamic_pressure"] = (
        n * mp * (v * 1e3) ** 2 * 1e9
    ).fillna(0.0)

    # ------------------------------------------------------------------
    # 5. IMF impulsiveness (turning / shocks)
    # ------------------------------------------------------------------
    working["delta_bz"] = bz.diff().fillna(0.0)

    # ------------------------------------------------------------------
    # Final NaN check (hard fail)
    # ------------------------------------------------------------------
    final_cols = required + [
        "clock_angle",
        "newell_dphi_dt",
        "ey",
        "dynamic_pressure",
        "delta_bz",
    ]

    missing = working[final_cols].isna().any()
    if missing.any():
        missing_cols = ", ".join(missing[missing].index.tolist())
        raise RuntimeError(
            f"NaNs detected after IMF/SW feature engineering in: {missing_cols}"
        )

    return working


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def engineer_sw_imf_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed IMF + solar wind dataset not found.")

    features = _add_sw_imf_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] IMF + solar wind engineered features saved to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_sw_imf_features()


if __name__ == "__main__":
    main()
