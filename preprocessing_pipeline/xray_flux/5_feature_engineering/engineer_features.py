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
# Configuration
# ---------------------------------------------------------------------
FLUX_COLS = ["irradiance_xrsa", "irradiance_xrsb"]
EPS = 1e-12
ROLLING_CHUNK = 20000


def _chunked_rolling(
    series: pd.Series,
    window: int,
    apply_fn,
) -> pd.Series:
    """Apply rolling operation in smaller chunks to avoid bottleneck issues."""
    if series.empty:
        return series.copy()

    pieces: list[pd.Series] = []
    n = len(series)
    for start in range(0, n, ROLLING_CHUNK):
        end = min(n, start + ROLLING_CHUNK)
        slice_start = max(0, start - window + 1)
        chunk = series.iloc[slice_start:end]
        rolled = apply_fn(chunk.rolling(window))
        # keep only the rows corresponding to the actual chunk range
        take = end - start
        pieces.append(rolled.iloc[-take:] if take else rolled.iloc[0:0])
    return pd.concat(pieces)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return _chunked_rolling(series, window, lambda roll: roll.mean())


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return _chunked_rolling(series, window, lambda roll: roll.std())


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    return _chunked_rolling(series, window, lambda roll: roll.max())


def _rolling_min(series: pd.Series, window: int) -> pd.Series:
    return _chunked_rolling(series, window, lambda roll: roll.min())


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    idx = np.arange(window, dtype=float)
    mean_x = idx.mean()
    denom = ((idx - mean_x) ** 2).sum()

    def _calc(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        mean_y = values.mean()
        return float(((values - mean_y) * (idx - mean_x)).sum() / denom)

    return _chunked_rolling(
        series,
        window,
        lambda roll: roll.apply(_calc, raw=True),
    )


# ---------------------------------------------------------------------
# Feature engineering (minimal, high-value)
# ---------------------------------------------------------------------
def _add_xrs_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()

    for col in FLUX_COLS + ["xrs_missing_flag"]:
        if col not in working.columns:
            raise RuntimeError(f"Required column '{col}' missing from dataset.")

    a = working["irradiance_xrsa"].astype(float)
    b = working["irradiance_xrsb"].astype(float)

    # --------------------------------------------------------------
    # Core log-space flux
    # --------------------------------------------------------------
    working["log_xrsa"] = np.log10(a + EPS)
    working["log_xrsb"] = np.log10(b + EPS)

    # --------------------------------------------------------------
    # Spectral hardness
    # --------------------------------------------------------------
    working["xrs_hardness"] = working["log_xrsb"] - working["log_xrsa"]

    # --------------------------------------------------------------
    # Temporal dynamics
    # --------------------------------------------------------------
    working["dlog_xrsb"] = working["log_xrsb"].diff()
    working["d2log_xrsb"] = working["dlog_xrsb"].diff()
    working["dlog_xrsa"] = working["log_xrsa"].diff()
    working["d2log_xrsa"] = working["dlog_xrsa"].diff()

    # --------------------------------------------------------------
    # Contextual statistics
    # --------------------------------------------------------------
    working["log_xrsb_mean_6h"] = _rolling_mean(working["log_xrsb"], 6)
    working["log_xrsb_std_6h"] = _rolling_std(working["log_xrsb"], 6)
    working["log_xrsb_max_24h"] = _rolling_max(working["log_xrsb"], 24)
    working["log_xrsb_slope_6h"] = _rolling_slope(working["log_xrsb"], 6)

    working["log_xrsa_mean_6h"] = _rolling_mean(working["log_xrsa"], 6)
    working["log_xrsa_std_6h"] = _rolling_std(working["log_xrsa"], 6)
    working["log_xrsa_max_24h"] = _rolling_max(working["log_xrsa"], 24)
    working["log_xrsa_slope_6h"] = _rolling_slope(working["log_xrsa"], 6)

    # --------------------------------------------------------------
    # Event morphology
    # --------------------------------------------------------------
    rolling_min_24h = _rolling_min(working["log_xrsb"], 24)
    working["peak_to_bg_24h_xrsb"] = working["log_xrsb"] - rolling_min_24h

    rolling_min_24h_xrsa = _rolling_min(working["log_xrsa"], 24)
    working["peak_to_bg_24h_xrsa"] = working["log_xrsa"] - rolling_min_24h_xrsa

    # --------------------------------------------------------------
    # Time since last rapid rise
    # --------------------------------------------------------------
    rapid_rise = working["dlog_xrsb"] > 0.3
    hrs_since = np.full(len(working), np.nan)

    last_event = None
    for i, flag in enumerate(rapid_rise):
        if flag:
            last_event = i
            hrs_since[i] = 0.0
        elif last_event is not None:
            hrs_since[i] = i - last_event

    working["hrs_since_rapid_rise_xrsb"] = hrs_since

    rapid_rise_a = working["dlog_xrsa"] > 0.3
    hrs_since_a = np.full(len(working), np.nan)
    last_event = None
    for i, flag in enumerate(rapid_rise_a):
        if flag:
            last_event = i
            hrs_since_a[i] = 0.0
        elif last_event is not None:
            hrs_since_a[i] = i - last_event
    working["hrs_since_rapid_rise_xrsa"] = hrs_since_a

    # --------------------------------------------------------------
    # Regime flags
    # --------------------------------------------------------------
    working["quiet_flag_xrsb"] = (working["log_xrsb"] < -7).astype(int)
    working["active_flag_xrsb"] = (
        (working["log_xrsb"] >= -7) & (working["log_xrsb"] < -5)
    ).astype(int)
    working["flaring_flag_xrsb"] = (working["log_xrsb"] >= -5).astype(int)

    working["quiet_flag_xrsa"] = (working["log_xrsa"] < -7).astype(int)
    working["active_flag_xrsa"] = (
        (working["log_xrsa"] >= -7) & (working["log_xrsa"] < -5)
    ).astype(int)
    working["flaring_flag_xrsa"] = (working["log_xrsa"] >= -5).astype(int)

    # --------------------------------------------------------------
    # Short memory
    # --------------------------------------------------------------
    working["log_xrsb_lag_1h"] = working["log_xrsb"].shift(1)
    working["log_xrsa_lag_1h"] = working["log_xrsa"].shift(1)

    # --------------------------------------------------------------
    # Final NaN handling and check
    # --------------------------------------------------------------
    engineered_cols = [c for c in working.columns if c not in df.columns]
    working[engineered_cols] = working[engineered_cols].fillna(0.0)

    missing = working[engineered_cols].isna().any()
    if missing.any():
        bad = ", ".join(missing[missing].index.tolist())
        raise RuntimeError(
            f"NaNs detected after XRS feature engineering in: {bad}"
        )

    return working


# ---------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------
def engineer_xrs_features() -> pd.DataFrame:
    df = load_hourly_output(IMPUTED_DB, IMPUTED_TABLE)
    if df.empty:
        raise RuntimeError("Imputed X-ray dataset not found.")

    features = _add_xrs_features(df)
    write_sqlite_table(features, OUTPUT_DB, OUTPUT_TABLE)

    print(f"[OK] X-ray engineered features saved to {OUTPUT_DB}")
    return features


def main() -> None:
    engineer_xrs_features()


if __name__ == "__main__":
    main()
