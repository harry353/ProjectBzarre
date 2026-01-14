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
import sqlite3

from preprocessing_pipeline.utils import load_hourly_output

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
SPLITS_DB = (
    STAGE_DIR.parents[1]
    / "imf_solar_wind"
    / "6_train_test_split"
    / "imf_solar_wind_imputed_split.db"
)
OUTPUT_DB = STAGE_DIR.parents[1] / "imf_solar_wind" / "imf_solar_wind_fin.db"
OUTPUT_TABLES = {
    "train": "imf_solar_wind_train",
    "validation": "imf_solar_wind_validation",
    "test": "imf_solar_wind_test",
}

MIN_FRACTION_COVERAGE = 0.5
BZ_MIN_WINDOW_H = 6
BZ_SOUTH_FRAC_WINDOW_H = 6
NEWELL_INT_WINDOW_H = 6
EY_MEAN_WINDOW_H = 6
PDYN_MAX_WINDOW_H = 6

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
def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bz = pd.Series(df["bz_gse"].to_numpy(dtype=float), index=df.index)
    newell = pd.Series(df["newell_dphi_dt"].to_numpy(dtype=float), index=df.index)
    ey = pd.Series(df["ey"].to_numpy(dtype=float), index=df.index)
    pdyn = pd.Series(df["dynamic_pressure"].to_numpy(dtype=float), index=df.index)

    def _min_periods(w: int) -> int:
        return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))

    w = BZ_MIN_WINDOW_H
    window = f"{w}h"
    out[f"bz_min_{w}h"] = (
        bz
        .rolling(window, min_periods=_min_periods(w))
        .min()
    )

    w = BZ_SOUTH_FRAC_WINDOW_H
    window = f"{w}h"
    out[f"bz_south_frac_{w}h"] = (
        (bz < 0)
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    w = NEWELL_INT_WINDOW_H
    window = f"{w}h"
    out[f"newell_int_{w}h"] = (
        newell
        .rolling(window, min_periods=_min_periods(w))
        .sum()
    )

    w = EY_MEAN_WINDOW_H
    window = f"{w}h"
    out[f"ey_mean_{w}h"] = (
        ey
        .rolling(window, min_periods=_min_periods(w))
        .mean()
    )

    w = PDYN_MAX_WINDOW_H
    window = f"{w}h"
    out[f"pdyn_max_{w}h"] = (
        pdyn
        .rolling(window, min_periods=_min_periods(w))
        .max()
    )

    out = out.dropna(subset=[
        "bz_gse",
        "newell_dphi_dt",
        "ey",
        "dynamic_pressure",
        f"bz_min_{BZ_MIN_WINDOW_H}h",
        f"bz_south_frac_{BZ_SOUTH_FRAC_WINDOW_H}h",
        f"newell_int_{NEWELL_INT_WINDOW_H}h",
        f"ey_mean_{EY_MEAN_WINDOW_H}h",
        f"pdyn_max_{PDYN_MAX_WINDOW_H}h",
    ])
    if out.empty:
        raise RuntimeError("No aggregated IMF + solar wind features produced.")

    return out


def engineer_sw_imf_features() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split, table in OUTPUT_TABLES.items():
        df = load_hourly_output(SPLITS_DB, table)
        if df.empty:
            raise RuntimeError("Imputed IMF + solar wind split not found; run split first.")

        features = _add_sw_imf_features(df)
        features = _build_agg(features)
        outputs[split] = features

        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[OK] IMF + solar wind engineered+aggregate features saved to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"Rows written ({split}): {len(features):,}")

    return outputs


def main() -> None:
    engineer_sw_imf_features()


if __name__ == "__main__":
    main()
