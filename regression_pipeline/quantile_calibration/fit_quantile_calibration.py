from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
LABELS_DB = PROJECT_ROOT / "preprocessing_pipeline" / "labels" / "dst_regression" / "dst_regression_labels.db"
DST_DB = PROJECT_ROOT / "preprocessing_pipeline" / "check_multicolinearity" / "all_preprocessed_sources.db"
MODEL_BASE = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"
CALIBRATION_DIR = PROJECT_ROOT / "regression_pipeline" / "quantile_calibration"

FEATURE_TABLES = {"validation": "pca_validation"}
LABEL_TABLES = {"validation": "dst_regression_validation"}
DST_TABLES = {"validation": "merged_validation"}

QUANTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)
DST_THRESHOLD = -20.0
MIN_UNIQUE = 5

np.random.seed(0)


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_localize(None)


def _prepare_validation(target: str) -> tuple[pd.DataFrame, list[str]]:
    feat = _load_table(FEATURES_DB, FEATURE_TABLES["validation"])
    labels = _load_table(LABELS_DB, LABEL_TABLES["validation"])[["timestamp", target]]
    dst = _load_table(DST_DB, DST_TABLES["validation"])[["timestamp", "dst_dst"]]

    feat["timestamp"] = _normalize_ts(feat["timestamp"])
    labels["timestamp"] = _normalize_ts(labels["timestamp"])
    dst["timestamp"] = _normalize_ts(dst["timestamp"])

    merged = feat.merge(dst, on="timestamp", how="inner").merge(labels, on="timestamp", how="inner")
    merged = merged.dropna(axis=0, how="any")
    if merged.empty:
        raise RuntimeError("Validation merge produced no rows.")

    feature_cols = [
        c for c in merged.columns
        if c != target and pd.api.types.is_numeric_dtype(merged[c])
    ]
    return merged, feature_cols


def _fit_isotonic(q_pred: np.ndarray, y_true: np.ndarray) -> IsotonicRegression | None:
    if q_pred.size < MIN_UNIQUE or np.unique(q_pred).size < MIN_UNIQUE:
        return None
    z = (y_true <= q_pred).astype(float)
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(q_pred, z)
    return calibrator


def _apply_calibration(q_pred: np.ndarray, calibrator: IsotonicRegression, tau: float) -> np.ndarray:
    p_hat = calibrator.predict(q_pred)
    eps = 1e-6
    scale = np.clip(tau / np.clip(p_hat, eps, 1.0), 0.5, 3.0)
    return q_pred + scale * (q_pred - np.median(q_pred))


def _coverage(y_true: np.ndarray, q_values: np.ndarray) -> float:
    return float(np.mean(y_true <= q_values))


def _regime_mask(values: pd.Series, regime: str) -> pd.Series:
    if regime == "storm":
        return values <= DST_THRESHOLD
    if regime == "calm":
        return values >= DST_THRESHOLD
    raise ValueError(f"Unknown regime: {regime}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_horizons() -> Iterable[int]:
    target_cols = [
        c for c in _load_table(LABELS_DB, LABEL_TABLES["validation"]).columns
        if c.startswith("h")
    ]
    return sorted(int(c[1:]) for c in target_cols if c[1:].isdigit())


def main() -> None:
    results: list[Dict] = []

    _ensure_dir(CALIBRATION_DIR)
    horizons = list(_iter_horizons())
    print(f"[INFO] Calibrating horizons: {horizons}")

    for h in horizons:
        target = f"h{h}"
        merged, feature_cols = _prepare_validation(target)

        for regime in ("storm", "calm"):
            df_reg = merged[_regime_mask(merged["dst_dst"], regime)].copy()
            if df_reg.empty:
                continue

            X = df_reg[feature_cols].to_numpy(dtype=np.float32)
            y_true = df_reg[target].to_numpy(dtype=np.float32)

            for q in QUANTILES:
                model_path = MODEL_BASE / f"h{h}_{regime}" / f"q{q}" / "model.joblib"
                if not model_path.exists():
                    continue

                model = joblib.load(model_path)
                q_pred = model.predict(X).astype(float)

                calibrator = _fit_isotonic(q_pred, y_true)
                if calibrator is None:
                    continue

                q_cal = _apply_calibration(q_pred, calibrator, q)
                before = _coverage(y_true, q_pred)
                after = _coverage(y_true, q_cal)

                out_dir = CALIBRATION_DIR / f"h{h}_{regime}" / f"q{q}"
                _ensure_dir(out_dir)
                joblib.dump(calibrator, out_dir / "calibrator.joblib")

                with open(out_dir / "metrics.json", "w") as f:
                    json.dump(
                        {
                            "horizon": h,
                            "regime": regime,
                            "quantile": q,
                            "n": int(len(y_true)),
                            "coverage_before": before,
                            "coverage_after": after,
                            "dst_threshold": DST_THRESHOLD,
                        },
                        f,
                        indent=2,
                    )

                results.append(
                    {
                        "horizon": h,
                        "regime": regime,
                        "quantile": q,
                        "coverage_before": before,
                        "coverage_after": after,
                        "n": int(len(y_true)),
                    }
                )

                print(
                    f"[OK] h{h} {regime} q{q}: "
                    f"coverage {before:.3f} -> {after:.3f}"
                )

    with open(CALIBRATION_DIR / "summary.json", "w") as f:
        json.dump(
            {"entries": results, "quantiles": QUANTILES, "dst_threshold": DST_THRESHOLD},
            f,
            indent=2,
        )

    print(f"[DONE] Wrote calibration artifacts to {CALIBRATION_DIR}")


if __name__ == "__main__":
    main()
