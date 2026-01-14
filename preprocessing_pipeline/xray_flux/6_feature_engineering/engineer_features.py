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

IMPUTED_DB = (
    STAGE_DIR.parents[1]
    / "xray_flux"
    / "5_train_test_split"
    / "xray_flux_imputed_split.db"
)
OUTPUT_DB = STAGE_DIR.parents[1] / "xray_flux" / "xray_flux_fin.db"
OUTPUT_TABLES = {
    "train": "xray_flux_train",
    "validation": "xray_flux_validation",
    "test": "xray_flux_test",
}

MIN_FRACTION_COVERAGE = 0.5
MEAN_WINDOW_H = 6
MAX_WINDOW_H = 24
SLOPE_WINDOW_H = 6
PEAK_BG_WINDOW_H = 24
RAPID_RISE_WINDOW_H = 24

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
def _min_periods(w: int) -> int:
    return max(1, int(np.ceil(w * MIN_FRACTION_COVERAGE)))


def _rolling_slope(series: pd.Series, w: int) -> pd.Series:
    def _calc(y: pd.Series) -> float:
        values = y.to_numpy(dtype=float)
        if np.isnan(values).any():
            return np.nan
        if len(values) < 2:
            return np.nan
        x = np.arange(len(values), dtype=float)
        xm = x.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0.0:
            return np.nan
        ym = values.mean()
        return float(((x - xm) * (values - ym)).sum() / denom)

    return series.rolling(f"{w}h", min_periods=_min_periods(w)).apply(_calc, raw=False)


def _build_agg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    xrsb = pd.Series(df["log_xrsb"].to_numpy(dtype=float), index=df.index)
    xrsa = pd.Series(df["log_xrsa"].to_numpy(dtype=float), index=df.index)
    dxrsb = pd.Series(df["dlog_xrsb_1h"].to_numpy(dtype=float), index=df.index)
    dxrsa = pd.Series(df["dlog_xrsa_1h"].to_numpy(dtype=float), index=df.index)

    w = MEAN_WINDOW_H
    window = f"{w}h"
    out[f"log_xrsb_mean_{w}h"] = xrsb.rolling(
        window, min_periods=_min_periods(w)
    ).mean()

    w = MAX_WINDOW_H
    window = f"{w}h"
    out[f"log_xrsb_max_{w}h"] = xrsb.rolling(
        window, min_periods=_min_periods(w)
    ).max()

    w = SLOPE_WINDOW_H
    out[f"log_xrsb_slope_{w}h"] = _rolling_slope(xrsb, w)

    w = PEAK_BG_WINDOW_H
    window = f"{w}h"
    bg = xrsb.rolling(window, min_periods=_min_periods(w)).min()
    out[f"log_xrsb_peak_to_bg_{w}h"] = xrsb - bg

    w = MEAN_WINDOW_H
    window = f"{w}h"
    out[f"log_xrsa_mean_{w}h"] = xrsa.rolling(
        window, min_periods=_min_periods(w)
    ).mean()

    w = MAX_WINDOW_H
    window = f"{w}h"
    out[f"log_xrsa_max_{w}h"] = xrsa.rolling(
        window, min_periods=_min_periods(w)
    ).max()

    w = SLOPE_WINDOW_H
    out[f"log_xrsa_slope_{w}h"] = _rolling_slope(xrsa, w)

    w = PEAK_BG_WINDOW_H
    window = f"{w}h"
    bg = xrsa.rolling(window, min_periods=_min_periods(w)).min()
    out[f"log_xrsa_peak_to_bg_{w}h"] = xrsa - bg

    w = RAPID_RISE_WINDOW_H
    window = f"{w}h"
    out[f"hrs_since_rapid_rise_xrsb_{w}h"] = (
        (dxrsb > 0.3)
        .astype(int)
        .rolling(window, min_periods=1)
        .apply(lambda x: np.argmax(x[::-1]) if x.any() else np.nan)
    )
    out[f"hrs_since_rapid_rise_xrsa_{w}h"] = (
        (dxrsa > 0.3)
        .astype(int)
        .rolling(window, min_periods=1)
        .apply(lambda x: np.argmax(x[::-1]) if x.any() else np.nan)
    )

    out = out.dropna(subset=[
        "log_xrsb",
        "log_xrsa",
        "dlog_xrsb_1h",
        "dlog_xrsa_1h",
        f"log_xrsb_mean_{MEAN_WINDOW_H}h",
        f"log_xrsb_max_{MAX_WINDOW_H}h",
        f"log_xrsb_slope_{SLOPE_WINDOW_H}h",
        f"log_xrsb_peak_to_bg_{PEAK_BG_WINDOW_H}h",
        f"log_xrsa_mean_{MEAN_WINDOW_H}h",
        f"log_xrsa_max_{MAX_WINDOW_H}h",
        f"log_xrsa_slope_{SLOPE_WINDOW_H}h",
        f"log_xrsa_peak_to_bg_{PEAK_BG_WINDOW_H}h",
        f"hrs_since_rapid_rise_xrsb_{RAPID_RISE_WINDOW_H}h",
        f"hrs_since_rapid_rise_xrsa_{RAPID_RISE_WINDOW_H}h",
    ])
    if out.empty:
        raise RuntimeError("No X-ray aggregate features produced.")

    return out


def engineer_xrs_features() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for split, table in OUTPUT_TABLES.items():
        df = load_hourly_output(IMPUTED_DB, table)
        if df.empty:
            raise RuntimeError("Imputed X-ray split not found; run split first.")

        features = _add_xrs_features(df)
        features = _build_agg(features)
        outputs[split] = features

        out = features.reset_index().rename(columns={features.index.name or "index": "timestamp"})
        with sqlite3.connect(OUTPUT_DB) as conn:
            out.to_sql(table, conn, if_exists="replace", index=False)

    print(f"[OK] X-ray engineered+aggregate features written to {OUTPUT_DB}")
    for split, features in outputs.items():
        print(f"Rows written ({split}): {len(features):,}")

    return outputs


def main() -> None:
    engineer_xrs_features()


if __name__ == "__main__":
    main()
