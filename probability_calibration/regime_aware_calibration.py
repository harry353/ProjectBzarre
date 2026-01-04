from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "validation_calibration.db"
TABLE_NAME = "validation"

CALIBRATOR_FILES = {
    "minimum": "calibrator_minimum.joblib",
    "ascending_maximum": "calibrator_ascending_maximum.joblib",
    "declining": "calibrator_declining.joblib",
}

HORIZONS = range(1, 9)

# ---------------------------------------------------------------------
# Regime definition (solar-cycle phase)
# ---------------------------------------------------------------------
REGIME_BINS = {
    "minimum": (0.0, 0.3),
    "ascending_maximum": (0.3, 0.7),
    "declining": (0.7, 1.0),
}

MIN_SAMPLES_PER_REGIME = 50

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_validation() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing calibration DB: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

    if df.empty:
        raise RuntimeError("Calibration dataset is empty.")

    required = {"timestamp", "solar_cycle_phase"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def _assign_regime(phase: float) -> str | None:
    if not np.isfinite(phase):
        return None
    for name, (start, end) in REGIME_BINS.items():
        if start <= phase < end or (name == "declining" and phase == 1.0):
            return name
    return None


def _fit_calibrator(df: pd.DataFrame, regime: str) -> IsotonicRegression:
    subset = df[df["regime"] == regime]

    if subset.empty:
        raise RuntimeError(f"No samples for regime '{regime}'.")

    if len(subset) < MIN_SAMPLES_PER_REGIME:
        raise RuntimeError(
            f"Insufficient samples for regime '{regime}': "
            f"{len(subset)} < {MIN_SAMPLES_PER_REGIME}"
        )

    x = pd.to_numeric(subset["raw_prob"], errors="coerce")
    y = pd.to_numeric(subset["label"], errors="coerce")

    mask = x.notna() & y.notna()
    x = x[mask].clip(0.0, 1.0).to_numpy()
    y = y[mask].astype(float).to_numpy()

    if len(x) < MIN_SAMPLES_PER_REGIME:
        raise RuntimeError(
            f"Insufficient valid samples for regime '{regime}': {len(x)}"
        )

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return model


# ---------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------
def calibrate_probability(
    raw_prob: float,
    solar_cycle_phase: float,
    horizon: int,
) -> float:
    regime = _assign_regime(float(solar_cycle_phase))
    if regime is None:
        raise ValueError("Invalid solar cycle phase.")

    calibration_dir = BASE_DIR / f"calibration_h{horizon}"
    model_path = calibration_dir / CALIBRATOR_FILES[regime]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing calibrator for regime '{regime}': {model_path}"
        )

    model = joblib.load(model_path)

    p = float(raw_prob)
    if not np.isfinite(p):
        raise ValueError("Raw probability is not finite.")

    p = min(max(p, 0.0), 1.0)
    return float(model.predict([p])[0])


# ---------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------
def main() -> None:
    df = _load_validation()
    for horizon in HORIZONS:
        prob_col = f"raw_prob_h{horizon}"
        label_col = f"label_h{horizon}"
        if prob_col not in df.columns or label_col not in df.columns:
            raise RuntimeError(
                f"Missing calibration columns for h{horizon}: {prob_col}, {label_col}"
            )

        horizon_df = df.copy()
        horizon_df["raw_prob"] = pd.to_numeric(horizon_df[prob_col], errors="coerce")
        horizon_df["label"] = pd.to_numeric(horizon_df[label_col], errors="coerce")
        horizon_df["solar_cycle_phase"] = pd.to_numeric(
            horizon_df["solar_cycle_phase"], errors="coerce"
        ).clip(0.0, 1.0)

        horizon_df = horizon_df.dropna(subset=["raw_prob", "label", "solar_cycle_phase"])
        horizon_df["regime"] = horizon_df["solar_cycle_phase"].apply(_assign_regime)
        horizon_df = horizon_df.dropna(subset=["regime"])

        if horizon_df.empty:
            raise RuntimeError(f"No valid calibration rows after cleaning for h{horizon}.")

        calibration_dir = BASE_DIR / f"calibration_h{horizon}"
        calibration_dir.mkdir(parents=True, exist_ok=True)

        calibrators: dict[str, str] = {}
        counts: dict[str, int] = {}

        for regime in REGIME_BINS:
            subset = horizon_df[horizon_df["regime"] == regime]
            counts[regime] = len(subset)

            model = _fit_calibrator(horizon_df, regime)
            fname = CALIBRATOR_FILES[regime]
            path = calibration_dir / fname
            joblib.dump(model, path)
            calibrators[regime] = fname

        metadata = {
            "regime_bins": REGIME_BINS,
            "min_samples_per_regime": MIN_SAMPLES_PER_REGIME,
            "sample_counts": counts,
            "calibrators": calibrators,
            "source_db": str(DB_PATH),
            "horizon": horizon,
        }

        metadata_path = calibration_dir / "calibration_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        print(f"[OK] Regime-aware calibration complete for h{horizon}")
        for r, n in counts.items():
            print(f"  {r:>20s}: {n:,} samples")


if __name__ == "__main__":
    main()
