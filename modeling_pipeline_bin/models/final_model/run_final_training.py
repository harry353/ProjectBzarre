from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple, List
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modeling_pipeline_bin.utils.data_loading_bin import load_split_dataframe


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PRUNED_FEATURES_PATH = PIPELINE_ROOT / "feature_pruning" / "pruned_features.json"
STAGE_A_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
STAGE_B_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
STAGE_C_PATH = PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "best_tree_params.json"
STAGE_D_PATH = PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "best_regularization.json"
STAGE_E_PATH = PIPELINE_ROOT / "optuna_studies" / "stageE_scale_pos_weight" / "best_scale_pos_weight.json"
ENSEMBLE_SUMMARY_PATH = PIPELINE_ROOT / "models" / "ensembles" / "ensemble_summary.json"

FINAL_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = FINAL_DIR / "outputs"

LOG_PATH = PIPELINE_ROOT / "logs" / "training" / "final_training.log"

HORIZON_CONSTANT = 4
HORIZONS = tuple(range(1, 9))
DEFAULT_N_JOBS = 12

# Decision policy config
ENABLE_WARNING_TIER = True
WARNING_MIN_PRECISION = 0.75
ENABLE_CONFIRMATION_GATE = True
CONFIRM_WINDOW = 2
CONFIRM_REQUIRED = 2
ENABLE_TREND_SUPPRESSION = True

LABEL_PREFIX = "label_not_quiet_h"
FUTURE_NORM_PREFIX = "dst_future_norm_h"
FUTURE_PHYS_PREFIX = "dst_future_physical_h"
TARGET_ENV = "PIPELINE_TARGET_COLUMN"
DEFAULT_TARGET_COLUMN = "severity_label"
TARGET_ALIAS = {
    "severity": "severity_label",
    "storm": "severity_label",
    "main": "main_phase_label",
    "main_phase": "main_phase_label",
    "ssc": "ssc_label",
}


def _resolve_target_column() -> str:
    token = os.environ.get(TARGET_ENV, DEFAULT_TARGET_COLUMN)
    if not token:
        return DEFAULT_TARGET_COLUMN
    return TARGET_ALIAS.get(token.lower(), token)



NON_FEATURE_COLS = {
    "timestamp",
    "time_tag",
}
EXCLUDED_PREFIXES = (
    LABEL_PREFIX,
    FUTURE_NORM_PREFIX,
    FUTURE_PHYS_PREFIX,
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_training")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _horizon_dir(horizon: int) -> Path:
    path = OUTPUT_ROOT / f"h{horizon}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_model(params: Dict[str, float], seed: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        gamma=float(params["gamma"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=seed,
        n_jobs=n_jobs,
        scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
    )


def _prepare_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    matrix = df[feature_cols]
    if matrix.isna().any().any():
        raise ValueError("NaNs detected in feature columns.")
    return matrix.to_numpy(dtype=np.float32)


def _get_labels(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise RuntimeError(f"Missing label column: {column}")
    series = df[column]
    if series.isna().any():
        raise ValueError(f"NaNs detected in label column: {column}")
    return series.to_numpy(dtype=np.int8)


def _select_warning_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    minimum_precision: float,
    base_threshold: float,
) -> float:
    best_threshold = base_threshold
    best_recall = -1.0
    thresholds = np.linspace(0.05, 0.95, 181)
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        if precision >= minimum_precision and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    return best_threshold


def _apply_confirmation_gate(
    predictions: np.ndarray,
    window: int,
    required: int,
) -> np.ndarray:
    if window <= 1 or required <= 1:
        return predictions.copy()
    series = pd.Series(predictions.astype(int))
    counts = series.rolling(window, min_periods=1).sum()
    return (counts >= required).to_numpy(dtype=bool)


def _compute_bz_mean(feature_df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in ("bz_gse", "bz_gse_lag_1h", "bz_gse_lag_2h") if c in feature_df.columns]
    if not candidates:
        return pd.Series(0.0, index=feature_df.index)
    return feature_df[candidates].mean(axis=1)


def _apply_trend_suppression(
    watch_preds: np.ndarray,
    warning_preds: np.ndarray,
    features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not ENABLE_TREND_SUPPRESSION:
        suppressed = np.zeros_like(watch_preds, dtype=bool)
        return suppressed, watch_preds.copy(), warning_preds.copy()

    dst_mean = features.get("dst_mean_6h")
    dst_current = features.get("dst")
    if dst_mean is None or dst_current is None:
        suppressed = np.zeros_like(watch_preds, dtype=bool)
        return suppressed, watch_preds.copy(), warning_preds.copy()

    mean_bz = _compute_bz_mean(features)

    increasing = dst_mean.diff().fillna(0.0) > 0
    dst_above_mean = dst_current > dst_mean
    bz_northward = mean_bz >= 0
    suppressed = (increasing & dst_above_mean & bz_northward).to_numpy(dtype=bool)
    final_watch = np.logical_and(watch_preds, ~suppressed)
    final_warning = np.logical_and(warning_preds, ~suppressed)
    return suppressed, final_watch, final_warning


def _compute_boolean_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    preds_int = preds.astype(int)
    return {
        "precision": float(precision_score(y_true, preds_int, zero_division=0)),
        "recall": float(recall_score(y_true, preds_int, zero_division=0)),
        "positive_rate": float(preds_int.mean()),
    }


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


# ---------------------------------------------------------------------
# Threshold selection (NEW)
# ---------------------------------------------------------------------
def _select_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    min_precision: float = 0.6,
) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_recall = -1.0

    for t in thresholds:
        pred = (prob >= t).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)

        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_t = t

    return float(best_t)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def _evaluate_predictions(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    pred_labels = (prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred_labels, average="binary", zero_division=0
    )

    metrics: Dict[str, float] = {
        "logloss": float(log_loss(y_true, prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, pred_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(pred_labels.mean()),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", HORIZON_CONSTANT))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    target_column = _resolve_target_column()
    logger = _setup_logger()
    logger.info("=== Final training started ===")
    if horizon_hours != HORIZON_CONSTANT:
        logger.info("Received horizon argument=%s (unused in multi-horizon training).", horizon_hours)
    logger.info("Using target column: %s", target_column)

    params = {}
    params.update(_load_json(STAGE_A_PATH))
    params.update(_load_json(STAGE_B_PATH))
    params.update(_load_json(STAGE_C_PATH))
    params.update(_load_json(STAGE_D_PATH))
    params.update({"scale_pos_weight": _load_json(STAGE_E_PATH).get("scale_pos_weight", 1.0)})

    pruned_features: List[str] = _load_json(PRUNED_FEATURES_PATH)
    summary = _load_json(ENSEMBLE_SUMMARY_PATH)

    feature_cols = [
        c
        for c in pruned_features
        if c not in NON_FEATURE_COLS
        and not any(c.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    ]

    strategy = summary["chosen_strategy"]
    seeds = summary.get("seeds", [])
    chosen_seed = summary.get("chosen_seed")

    for horizon in HORIZONS:
        X_train_df, y_train_series = load_split_dataframe("train", horizon)
        X_val_df, y_val_series = load_split_dataframe("validation", horizon)
        X_test_df, y_test_series = load_split_dataframe("test", horizon)

        usable_features = [
            c for c in feature_cols if c in X_train_df.columns
        ]
        missing = [c for c in feature_cols if c not in X_train_df.columns]
        if missing:
            logger.warning("H%d: dropping %d missing features: %s", horizon, len(missing), missing[:5])

        X_train = _prepare_features(X_train_df, usable_features)
        X_val = _prepare_features(X_val_df, usable_features)
        X_test = _prepare_features(X_test_df, usable_features)

        y_train = (y_train_series > 0).to_numpy(dtype=np.int8)
        y_val = (y_val_series > 0).to_numpy(dtype=np.int8)
        y_test = (y_test_series > 0).to_numpy(dtype=np.int8)

        future_norm_col = f"{FUTURE_NORM_PREFIX}{horizon}"
        future_phys_col = f"{FUTURE_PHYS_PREFIX}{horizon}"
        X_train_final = np.vstack([X_train, X_val])
        y_train_final = np.concatenate([y_train, y_val])

        logger.info("=== Horizon h%d ===", horizon)
        base_model = _build_model(params, seed=1337, n_jobs=n_jobs)
        base_model.fit(X_train, y_train, verbose=False)
        val_prob = base_model.predict_proba(X_val)[:, 1]
        watch_threshold = _select_threshold(y_val, val_prob)
        if ENABLE_WARNING_TIER:
            warning_threshold = _select_warning_threshold(
                val_prob,
                y_val,
                WARNING_MIN_PRECISION,
                watch_threshold,
            )
        else:
            warning_threshold = watch_threshold
        print(
            f"[INFO] Horizon h{horizon}: watch threshold={watch_threshold:.3f} "
            f"warning threshold={warning_threshold:.3f}"
        )

        h_dir = _horizon_dir(horizon)

        if strategy == "single":
            if chosen_seed is None:
                raise ValueError("Chosen seed missing in ensemble summary for single strategy.")
            model = _build_model(params, seed=chosen_seed, n_jobs=n_jobs)
            model.fit(X_train_final, y_train_final, verbose=False)
            prob = model.predict_proba(X_test)[:, 1]
            model_path = h_dir / "final_model_single.json"
            model.get_booster().save_model(model_path)
            ensemble_info = {
                "strategy": "single",
                "seed": chosen_seed,
                "model_path": str(model_path),
            }
        elif strategy == "ensemble":
            if not seeds:
                raise ValueError("Ensemble strategy requires 'seeds' in ensemble summary.")
            preds: List[np.ndarray] = []
            model_paths: List[str] = []
            for seed in seeds:
                model = _build_model(params, seed=seed, n_jobs=n_jobs)
                model.fit(X_train_final, y_train_final, verbose=False)
                preds.append(model.predict_proba(X_test)[:, 1])
                path = h_dir / f"final_model_seed_{seed}.json"
                model.get_booster().save_model(path)
                model_paths.append(str(path))
            prob = np.mean(preds, axis=0)
            ensemble_info = {
                "strategy": "ensemble",
                "seeds": seeds,
                "model_paths": model_paths,
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        metrics = _evaluate_predictions(y_test, prob, watch_threshold)

        metrics_path = h_dir / "final_test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        threshold_path = h_dir / "decision_threshold.json"
        threshold_payload = {
            "label": f"{target_column}_binary",
            "horizon_hours": horizon,
            "watch_threshold": watch_threshold,
            "warning_threshold": warning_threshold,
        }
        threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")

        ensemble_info_path = h_dir / "final_ensemble_info.json"
        ensemble_info_path.write_text(json.dumps(ensemble_info, indent=2), encoding="utf-8")

        raw_watch = (prob >= watch_threshold)
        raw_warning = (prob >= warning_threshold)

        if ENABLE_CONFIRMATION_GATE:
            watch_confirmed = _apply_confirmation_gate(raw_watch, CONFIRM_WINDOW, CONFIRM_REQUIRED)
            warning_confirmed = _apply_confirmation_gate(raw_warning, CONFIRM_WINDOW, CONFIRM_REQUIRED)
        else:
            watch_confirmed = raw_watch.copy()
            warning_confirmed = raw_warning.copy()

        suppressed_by_trend, final_watch, final_warning = _apply_trend_suppression(
            watch_confirmed, warning_confirmed, X_test_df
        )

        policy_metrics = {
            "watch_raw": _compute_boolean_metrics(y_test, raw_watch),
            "warning_raw": _compute_boolean_metrics(y_test, raw_warning),
            "watch_confirmed": _compute_boolean_metrics(y_test, watch_confirmed),
            "warning_confirmed": _compute_boolean_metrics(y_test, warning_confirmed),
            "watch_final": _compute_boolean_metrics(y_test, final_watch),
            "warning_final": _compute_boolean_metrics(y_test, final_warning),
        }

        predictions_path = h_dir / "final_test_predictions.parquet"
        actual_label_col = f"actual_{target_column}"
        predictions_df = pd.DataFrame(
            {
                "prob_not_quiet": prob,
                "pred_label": raw_watch.astype(int),
                "actual_label": y_test,
                actual_label_col: y_test_series.to_numpy(dtype=np.int8),
                "pred_watch": raw_watch.astype(int),
                "pred_warning": raw_warning.astype(int),
                "watch_confirmed": watch_confirmed.astype(int),
                "warning_confirmed": warning_confirmed.astype(int),
                "suppressed_by_trend": suppressed_by_trend.astype(bool),
                "final_watch": final_watch.astype(int),
                "final_warning": final_warning.astype(int),
                future_norm_col: X_test_df[future_norm_col].to_numpy(dtype=float)
                if future_norm_col in X_test_df.columns
                else np.full_like(prob, np.nan, dtype=float),
                future_phys_col: X_test_df[future_phys_col].to_numpy(dtype=float)
                if future_phys_col in X_test_df.columns
                else np.full_like(prob, np.nan, dtype=float),
            }
        )
        if target_column == "severity_label":
            predictions_df["actual_severity_class"] = predictions_df[actual_label_col]
        predictions_df.to_parquet(predictions_path, index=False)

        policy_metrics_path = h_dir / "final_policy_metrics.json"
        policy_payload = {
            "horizon_hours": horizon,
            "metrics": policy_metrics,
        }
        policy_metrics_path.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

        print(f"[INFO] Horizon h{horizon} metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
        logger.info("Horizon h%d metrics: %s", horizon, metrics)
        logger.info("Horizon h%d thresholds watch=%.4f warning=%.4f", horizon, watch_threshold, warning_threshold)

    logger.info("=== Final training complete for horizons %s ===", HORIZONS)


if __name__ == "__main__":
    main()
