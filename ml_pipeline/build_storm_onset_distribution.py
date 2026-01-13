from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZON_ROOT = PROJECT_ROOT / "ml_pipeline" / "horizon_models"
OUTPUT_DB = PROJECT_ROOT / "ml_pipeline" / "storm_onset_distribution.db"
OUTPUT_TABLE = "storm_onset_distribution"
HORIZONS = range(1, 9)


def _load_horizon(horizon: int) -> pd.DataFrame:
    db_path = HORIZON_ROOT / f"h{horizon}" / "calibrated_probabilities.db"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing calibrated DB for h{horizon}: {db_path}")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT timestamp, prob_calibrated FROM calibrated_probs",
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "prob_calibrated"])
    df = df.rename(columns={"prob_calibrated": f"p_{horizon}"})
    return df


def main() -> None:
    merged = None
    for h in HORIZONS:
        df = _load_horizon(h)
        merged = df if merged is None else merged.merge(df, on="timestamp", how="inner")

    if merged is None or merged.empty:
        raise RuntimeError("No overlapping timestamps across horizons.")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    p_cols = [f"p_{h}" for h in HORIZONS]
    probs = merged[p_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    probs = np.maximum.accumulate(probs, axis=1)

    intervals = np.zeros_like(probs)
    intervals[:, 0] = probs[:, 0]
    intervals[:, 1:] = probs[:, 1:] - probs[:, :-1]
    intervals = np.clip(intervals, 0.0, 1.0)
    survival = np.clip(1.0 - probs[:, -1], 0.0, 1.0)

    out = pd.DataFrame({"timestamp": merged["timestamp"]})
    for idx, h in enumerate(HORIZONS):
        out[f"p_h{h}"] = intervals[:, idx]
    out["p_survival_8h"] = survival

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as conn:
        out.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
