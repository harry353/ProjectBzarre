from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modeling_pipeline_bin.utils.data_loading_bin import (
    DEFAULT_HORIZON_HOURS,
    load_split_arrays,
    load_split_dataframe,
)

STAGE_A_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
STAGE_B_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
STAGE_C_PATH = PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "best_tree_params.json"
STAGE_D_PATH = PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "best_regularization.json"
STAGE_E_PATH = PIPELINE_ROOT / "optuna_studies" / "stageE_scale_pos_weight" / "best_scale_pos_weight.json"
PRUNED_FEATURES_PATH = Path(__file__).resolve().parent / "pruned_features.json"
IMPORTANCE_HISTORY_PATH = Path(__file__).resolve().parent / "feature_importance_history.json"
LOG_PATH = PIPELINE_ROOT / "logs" / "pruning" / "pruning_steps.log"
DEFAULT_N_JOBS = 12

MIN_FEATURE_THRESHOLD = 30
REMOVAL_FRACTION = 0.05
PROTECTED_PREFIXES = [
    "Dst",
    "dst",
    "Dst_lag",
    "dst_lag",
    "bz",
    "by",
    "bt",
    "speed",
    "newell",
    "epsilon",
    "imf_",
    "dynamic_pressure",
    "log_dynamic_pressure",
]


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("feature_pruning")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required parameter file missing: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_stage_params() -> dict:
    stage_a = _load_json(STAGE_A_PATH)
    stage_b = _load_json(STAGE_B_PATH)
    stage_c = _load_json(STAGE_C_PATH)
    stage_d = _load_json(STAGE_D_PATH)
    stage_e = _load_json(STAGE_E_PATH)

    params = {
        "learning_rate": stage_a["learning_rate"],
        "n_estimators": stage_b["n_estimators"],
        "max_depth": int(stage_d.get("max_depth", stage_c["max_depth"])),
        "min_child_weight": stage_d.get("min_child_weight", stage_c["min_child_weight"]),
        "gamma": stage_d.get("gamma", stage_c["gamma"]),
        "subsample": stage_d.get("subsample", stage_c["subsample"]),
        "colsample_bytree": stage_d.get("colsample_bytree", stage_c["colsample_bytree"]),
        "reg_alpha": stage_d.get("reg_alpha") or stage_d.get("alpha", 0.0),
        "reg_lambda": stage_d.get("reg_lambda") or stage_d.get("lambda", 1.0),
        "scale_pos_weight": stage_e.get("scale_pos_weight", 1.0),
    }
    return params


def _compute_importance(model: XGBClassifier, X_val: np.ndarray, feature_names: Sequence[str]) -> Dict[str, float]:
    booster = model.get_booster()
    dval = xgb.DMatrix(X_val, feature_names=list(feature_names))
    contribs = booster.predict(dval, pred_contribs=True)
    contribs = contribs[:, :-1]  # Drop bias term
    scores = np.mean(np.abs(contribs), axis=0)
    return {
        name: float(np.asarray(score).item() if np.asarray(score).size == 1 else np.mean(score))
        for name, score in zip(feature_names, scores)
    }


def _select_features_to_drop(importances: Dict[str, float], removal_count: int) -> List[str]:
    sorted_feats = sorted(importances.items(), key=lambda kv: kv[1])
    to_remove: List[str] = []
    for feat, _ in sorted_feats:
        if any(feat.startswith(prefix) for prefix in PROTECTED_PREFIXES):
            continue
        to_remove.append(feat)
        if len(to_remove) >= removal_count:
            break
    return to_remove


def _build_model(params: dict, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        learning_rate=params["learning_rate"],
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        gamma=float(params["gamma"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=1337,
        eval_metric="logloss",
        scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
        n_jobs=n_jobs,
    )


def main() -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", DEFAULT_HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Feature Pruning Started ===")
    print("[INFO] Loading stage parameters...")
    params = _load_stage_params()
    print(f"[INFO] Stage parameters: {params}")

    print(f"[INFO] Loading datasets (horizon={horizon_hours}h)...")
    train_df, _ = load_split_dataframe("train", horizon_hours)
    feature_names = train_df.columns.tolist()
    X_train, y_train = load_split_arrays("train", horizon_hours)
    X_val, y_val = load_split_arrays("validation", horizon_hours)
    logger.info(
        "Datasets ready: horizon=%sh, train=%s, val=%s",
        horizon_hours,
        X_train.shape,
        X_val.shape,
    )
    current_features = feature_names.copy()

    best_val_logloss = float("inf")
    best_features = current_features.copy()
    history: List[dict] = []
    iteration = 0

    while len(current_features) > MIN_FEATURE_THRESHOLD:
        iteration += 1
        print(f"[INFO] Iteration {iteration}: training with {len(current_features)} features")
        logger.info("Iteration %d: feature_count=%d", iteration, len(current_features))
        model = _build_model(params, n_jobs)
        feature_idx = [feature_names.index(f) for f in current_features]
        model.fit(
            X_train[:, feature_idx],
            y_train,
            eval_set=[(X_val[:, feature_idx], y_val)],
            verbose=False,
        )
        val_prob = model.predict_proba(X_val[:, feature_idx])[:, 1]
        val_logloss = float(log_loss(y_val, val_prob, labels=[0, 1]))
        print(f"[INFO] Validation logloss: {val_logloss:.6f}")
        logger.info("Iteration %d: val_logloss=%.6f", iteration, val_logloss)

        importances = _compute_importance(model, X_val[:, feature_idx], current_features)
        ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        top_20 = ranked[:20]

        history.append(
            {
                "iteration": iteration,
                "feature_count": len(current_features),
                "val_logloss": val_logloss,
                "top_20": [{"feature": f, "importance": float(score)} for f, score in top_20],
                "full_rank": [{"feature": f, "importance": float(score)} for f, score in ranked],
            }
        )

        if val_logloss < best_val_logloss - 1e-6:
            best_val_logloss = val_logloss
            best_features = current_features.copy()
        else:
            print("[INFO] Validation logloss did not improve; stopping pruning.")
            logger.info("Stopping pruning due to no improvement.")
            break

        removal_count = max(1, int(len(current_features) * REMOVAL_FRACTION))
        to_remove = _select_features_to_drop(importances, removal_count)
        if not to_remove:
            print("[INFO] No removable features found; stopping pruning.")
            logger.info("No removable features; stopping.")
            break

        logger.info("Removing %d features: %s", len(to_remove), to_remove)
        print(f"[INFO] Removing {len(to_remove)} features")
        current_features = [feat for feat in current_features if feat not in to_remove]

    PRUNED_FEATURES_PATH.write_text(
        json.dumps(best_features, indent=2),
        encoding="utf-8",
    )
    IMPORTANCE_HISTORY_PATH.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    print("[INFO] Feature pruning complete.")
    print(f"[INFO] Iterations performed: {len(history)}")
    print(f"[INFO] Initial features: {len(feature_names)} -> Final: {len(best_features)}")
    print(f"[INFO] Best validation logloss: {best_val_logloss:.6f}")
    logger.info(
        "Pruning complete: iterations=%d initial=%d final=%d best_val_logloss=%.6f",
        len(history),
        len(feature_names),
        len(best_features),
        best_val_logloss,
    )


if __name__ == "__main__":
    main()
