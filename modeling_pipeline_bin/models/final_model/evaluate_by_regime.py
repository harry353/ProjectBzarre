from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support

# ---------------------------------------------------------------------
# Paths (RELATIVE, robust)
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

PRED_PATH = HERE / "final_test_predictions.parquet"
OUT_PATH = HERE / "final_test_metrics_by_regime.json"

# ---------------------------------------------------------------------
# Column names (match your parquet schema)
# ---------------------------------------------------------------------
LABEL_COL = "actual_label"
PROB_COL = "prob_not_quiet"
PRED_LABEL_COL = "pred_label"
DST_PHYS_COL = "dst_future_physical_h4"

# ---------------------------------------------------------------------
# Regime thresholds (PHYSICAL UNITS, nT)
# ---------------------------------------------------------------------
QUIET_THR = -20.0
STORM_THR = -50.0


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def dst_regime(dst: np.ndarray) -> np.ndarray:
    """
    Classify Dst into quiet / moderate / storm using PHYSICAL units.
    """
    out = np.full(dst.shape, "quiet", dtype=object)
    out[dst <= QUIET_THR] = "moderate"
    out[dst <= STORM_THR] = "storm"
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

    df = pd.read_parquet(PRED_PATH)

    required_cols = {LABEL_COL, PROB_COL, DST_PHYS_COL, PRED_LABEL_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Prediction file missing required columns: {missing}")

    y_true = df[LABEL_COL].to_numpy(dtype=int)
    prob = df[PROB_COL].to_numpy(dtype=float)
    pred_label = df[PRED_LABEL_COL].to_numpy(dtype=int)
    dst_phys = df[DST_PHYS_COL].to_numpy(dtype=float)

    regimes = dst_regime(dst_phys)

    results: dict[str, dict] = {}

    for name in ("quiet", "moderate", "storm"):
        mask = regimes == name
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = prob[mask]
        yhat = pred_label[mask]

        if yt.size == 0:
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            yt, yhat, average="binary", zero_division=0
        )

        regime_metrics = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yhat)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "logloss": float(log_loss(yt, yp, labels=[0, 1])),
            "mean_probability": float(np.mean(yp)),
        }
        results[name] = regime_metrics

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------
    print("\nDst regime-based classification evaluation (TEST SET)")
    print("-" * 72)

    for regime, m in results.items():
        print(
            f"{regime.upper():8s} | "
            f"N={m['count']:6d} | "
            f"ACC={m['accuracy']:.3f} | "
            f"P={m['precision']:.3f} | "
            f"R={m['recall']:.3f} | "
            f"F1={m['f1']:.3f} | "
            f"logloss={m['logloss']:.4f}"
        )

    # --------------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------------
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Regime metrics written to {OUT_PATH}")


if __name__ == "__main__":
    main()
