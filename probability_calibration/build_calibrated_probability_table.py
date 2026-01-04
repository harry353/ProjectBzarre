from __future__ import annotations

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DB = BASE_DIR / "validation_calibration.db"
CALIBRATION_TABLE = "validation"

OUTPUT_DB = BASE_DIR / "calibrated_probabilities.db"
OUTPUT_TABLE = "calibrated_probabilities"

HORIZONS = range(1, 9)

CALIBRATOR_FILES = {
    "minimum": "calibrator_minimum.joblib",
    "ascending_maximum": "calibrator_ascending_maximum.joblib",
    "declining": "calibrator_declining.joblib",
}

REGIME_BINS = {
    "minimum": (0.0, 0.3),
    "ascending_maximum": (0.3, 0.7),
    "declining": (0.7, 1.0),
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _assign_regime(phase: float) -> str | None:
    if not np.isfinite(phase):
        return None
    for name, (lo, hi) in REGIME_BINS.items():
        if lo <= phase < hi or (name == "declining" and phase == 1.0):
            return name
    return None


def _load_calibrators() -> dict[int, dict[str, object]]:
    """
    Load all calibrators into memory:
      calibrators[horizon][regime] -> isotonic model
    """
    calibrators: dict[int, dict[str, object]] = {}

    for h in HORIZONS:
        h_dir = BASE_DIR / f"calibration_h{h}"
        if not h_dir.exists():
            raise FileNotFoundError(f"Missing calibration directory: {h_dir}")

        calibrators[h] = {}
        for regime, fname in CALIBRATOR_FILES.items():
            path = h_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing calibrator: {path}")
            calibrators[h][regime] = joblib.load(path)

    return calibrators


def _load_raw_probs() -> pd.DataFrame:
    if not CALIBRATION_DB.exists():
        raise FileNotFoundError(f"Missing calibration DB: {CALIBRATION_DB}")

    with sqlite3.connect(CALIBRATION_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {CALIBRATION_TABLE}", conn)

    if df.empty:
        raise RuntimeError("Calibration dataset is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["solar_cycle_phase"] = pd.to_numeric(
        df["solar_cycle_phase"], errors="coerce"
    ).clip(0.0, 1.0)

    df = df.dropna(subset=["timestamp", "solar_cycle_phase"])
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    df = _load_raw_probs()
    calibrators = _load_calibrators()

    df["regime"] = df["solar_cycle_phase"].apply(_assign_regime)
    df = df.dropna(subset=["regime"])

    if df.empty:
        raise RuntimeError("No rows left after regime assignment.")

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "solar_cycle_phase": df["solar_cycle_phase"],
            "regime": df["regime"],
        }
    )

    for h in HORIZONS:
        raw_col = f"raw_prob_h{h}"
        if raw_col not in df.columns:
            raise RuntimeError(f"Missing column: {raw_col}")

        raw = pd.to_numeric(df[raw_col], errors="coerce").clip(0.0, 1.0)
        calibrated = np.full(len(df), np.nan, dtype=float)

        for regime, model in calibrators[h].items():
            mask = df["regime"] == regime
            if not mask.any():
                continue
            calibrated[mask] = model.predict(raw[mask].to_numpy())

        out[f"p_storm_h{h}"] = calibrated

    # Final sanity
    prob_cols = [f"p_storm_h{h}" for h in HORIZONS]
    out = out.dropna(subset=prob_cols)

    if out.empty:
        raise RuntimeError("No calibrated probabilities produced.")

    with sqlite3.connect(OUTPUT_DB) as conn:
        out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] Calibrated probability table written")
    print(f"     Rows     : {len(out):,}")
    print(f"     Horizons : h1–h8")
    print(f"     DB       : {OUTPUT_DB}")
    print(f"     Table    : {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
