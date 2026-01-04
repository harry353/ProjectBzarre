from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

INPUT_DB = BASE_DIR / "calibrated_probabilities.db"
INPUT_TABLE = "calibrated_probabilities"

OUTPUT_DB = BASE_DIR / "survival_probabilities.db"
OUTPUT_TABLE = "survival_probabilities"

HORIZONS = range(1, 9)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_calibrated() -> pd.DataFrame:
    if not INPUT_DB.exists():
        raise FileNotFoundError(f"Missing calibrated DB: {INPUT_DB}")

    with sqlite3.connect(INPUT_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {INPUT_TABLE}", conn)

    if df.empty:
        raise RuntimeError("Calibrated probability table is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    required = {f"p_storm_h{h}" for h in HORIZONS}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing probability columns: {sorted(missing)}")

    return df


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def _compute_survival(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat p_storm_hk as discrete hazard probabilities:
      h_k = P(storm starts in hour k | no storm before)

    Survival:
      S_k = Π_{i=1..k} (1 - h_i)

    CDF:
      F_k = 1 - S_k
    """
    hazards = np.vstack(
        [df[f"p_storm_h{h}"].to_numpy(dtype=float) for h in HORIZONS]
    ).T

    hazards = np.clip(hazards, 0.0, 1.0)

    survival = np.cumprod(1.0 - hazards, axis=1)
    cdf = 1.0 - survival

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "solar_cycle_phase": df.get("solar_cycle_phase"),
            "regime": df.get("regime"),
        }
    )

    for i, h in enumerate(HORIZONS):
        out[f"survival_to_h{h}"] = survival[:, i]
        out[f"p_storm_within_{h}h"] = cdf[:, i]

    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    df = _load_calibrated()
    out = _compute_survival(df)

    if out.empty:
        raise RuntimeError("No survival probabilities computed.")

    with sqlite3.connect(OUTPUT_DB) as conn:
        out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)

    print("[OK] Survival probabilities computed")
    print(f"     Rows     : {len(out):,}")
    print(f"     Horizons : 1–8 h")
    print(f"     DB       : {OUTPUT_DB}")
    print(f"     Table    : {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
