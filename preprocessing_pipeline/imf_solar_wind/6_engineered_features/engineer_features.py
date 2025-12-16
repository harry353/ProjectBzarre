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

    for col in ESSENTIAL_COLUMNS:
        if col not in working.columns:
            raise RuntimeError(f"Required column '{col}' missing from imputed dataset.")

    bx = working["bx_gse"].astype(float)
    by = working["by_gse"].astype(float)
    bz = working["bz_gse"].astype(float)
    bt = working["bt"].astype(float)
    v = working["speed"].astype(float)
    n = working["density"].astype(float)

    # --------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------
    clock = np.arctan2(by, bz)
    sin_h = np.sin(clock / 2.0)

    working["clock_angle"] = clock.fillna(0.0)
    working["sin_half_clock"] = sin_h.fillna(0.0)
    working["sin_half_clock_sq"] = (sin_h ** 2).fillna(0.0)
    working["sin_half_clock_4"] = (sin_h ** 4).fillna(0.0)
    working["sin_half_clock_8_3"] = (np.abs(sin_h) ** (8.0 / 3.0)).fillna(0.0)

    # --------------------------------------------------------------
    # Southward IMF
    # --------------------------------------------------------------
    bz_s = np.minimum(bz, 0.0)
    working["bz_south"] = bz_s.fillna(0.0)

    # --------------------------------------------------------------
    # Electric field proxies
    # --------------------------------------------------------------
    working["ey"] = (-v * bz).fillna(0.0)
    working["vbs"] = (v * np.abs(bz_s)).fillna(0.0)
    working["vbs_squared"] = (v * bz_s ** 2).fillna(0.0)

    # --------------------------------------------------------------
    # Dynamic pressure (nPa)
    # --------------------------------------------------------------
    mp = 1.6726219e-27
    pdyn = n * mp * (v * 1e3) ** 2 * 1e9
    working["dynamic_pressure"] = pdyn.fillna(0.0)
    working["pd_sqrt"] = np.sqrt(np.maximum(pdyn, 0.0)).fillna(0.0)

    # --------------------------------------------------------------
    # Coupling functions
    # --------------------------------------------------------------
    working["epsilon"] = (
        v * bt ** 2 * working["sin_half_clock_4"]
    ).fillna(0.0)

    working["newell_dphi_dt"] = (
        (v ** (4.0 / 3.0))
        * (bt ** (2.0 / 3.0))
        * working["sin_half_clock_8_3"]
    ).fillna(0.0)

    working["kan_lee_efield"] = (
        v * bt * working["sin_half_clock_sq"]
    ).fillna(0.0)

    working["boyle_index"] = (
        1e-4 * v ** 2 + 11.7 * bt * (working["sin_half_clock"] ** 3)
    ).fillna(0.0)

    # --------------------------------------------------------------
    # Impulsiveness
    # --------------------------------------------------------------
    working["delta_bz"] = bz.diff().fillna(0.0)
    working["delta_bt"] = bt.diff().fillna(0.0)
    working["delta_speed"] = v.diff().fillna(0.0)

    # --------------------------------------------------------------
    # Regime flags
    # --------------------------------------------------------------
    working["southward_flag"] = (bz < 0).astype(int)
    working["high_speed_flag"] = (v >= 500).astype(int)

    # --------------------------------------------------------------
    # Short-term lags (1–3 hours), NaN-safe
    # --------------------------------------------------------------
    LAG_HOURS = [1, 2, 3]
    lag_sources = [
        "bx_gse",
        "by_gse",
        "bz_gse",
        "bt",
        "speed",
        "density",
        "vbs",
        "epsilon",
        "newell_dphi_dt",
    ]

    for col in lag_sources:
        for lag in LAG_HOURS:
            working[f"{col}_lag_{lag}h"] = (
                working[col].shift(lag).fillna(0.0)
            )

    # --------------------------------------------------------------
    # Final NaN check (hard fail)
    # --------------------------------------------------------------
    engineered_cols = [
        "clock_angle",
        "sin_half_clock",
        "sin_half_clock_sq",
        "sin_half_clock_4",
        "sin_half_clock_8_3",
        "bz_south",
        "ey",
        "vbs",
        "vbs_squared",
        "dynamic_pressure",
        "pd_sqrt",
        "epsilon",
        "newell_dphi_dt",
        "kan_lee_efield",
        "boyle_index",
        "delta_bz",
        "delta_bt",
        "delta_speed",
        "southward_flag",
        "high_speed_flag",
    ]

    lag_cols = [
        f"{col}_lag_{lag}h"
        for col in lag_sources
        for lag in LAG_HOURS
    ]

    to_check = ESSENTIAL_COLUMNS + engineered_cols + lag_cols
    missing = working[to_check].isna().any()
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
