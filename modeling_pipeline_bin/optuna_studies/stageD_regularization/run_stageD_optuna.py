from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple
import sys

import numpy as np
import optuna
from sklearn.metrics import log_loss
from optuna.trial import Trial
from xgboost import XGBClassifier

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modeling_pipeline_bin.utils.data_loading_bin import (
    DEFAULT_HORIZON_HOURS,
    load_split_arrays,
)

BEST_LR_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
BEST_ITERS_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
BEST_TREE_PATH = PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "best_tree_params.json"
LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageD.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_regularization.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageD_regularization"
DEFAULT_TRIALS = 80
HORIZON_HOURS = DEFAULT_HORIZON_HOURS
DEFAULT_N_JOBS = 12


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageD")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_stage_params() -> Tuple[float, dict, dict]:
    if not BEST_LR_PATH.exists():
        raise FileNotFoundError(f"Stage A best_lr.json not found at {BEST_LR_PATH}")
    with BEST_LR_PATH.open("r", encoding="utf-8") as fp:
        stage_a = json.load(fp)
    lr = stage_a.get("learning_rate")
    if lr is None:
        raise ValueError("learning_rate missing in Stage A output.")

    if not BEST_ITERS_PATH.exists():
        raise FileNotFoundError(f"Stage B best_iters.json not found at {BEST_ITERS_PATH}")
    with BEST_ITERS_PATH.open("r", encoding="utf-8") as fp:
        stage_b = json.load(fp)
    for key in ["n_estimators"]:
        if key not in stage_b:
            raise ValueError(f"{key} missing in Stage B output.")

    if not BEST_TREE_PATH.exists():
        raise FileNotFoundError(f"Stage C best_tree_params.json not found at {BEST_TREE_PATH}")
    with BEST_TREE_PATH.open("r", encoding="utf-8") as fp:
        stage_c = json.load(fp)
    for key in ["max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree"]:
        if key not in stage_c:
            raise ValueError(f"{key} missing in Stage C output.")

    return float(lr), stage_b, stage_c


def _prepare_data(horizon_hours: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = load_split_arrays("train", horizon_hours)
    X_val, y_val = load_split_arrays("validation", horizon_hours)

    train_inf_cols = np.unique(np.where(~np.isfinite(X_train))[1]).tolist()
    val_inf_cols = np.unique(np.where(~np.isfinite(X_val))[1]).tolist()
    if train_inf_cols:
        raise ValueError(f"Infinite values detected in training feature columns at indices: {train_inf_cols}")
    if val_inf_cols:
        raise ValueError(f"Infinite values detected in validation feature columns at indices: {val_inf_cols}")

    return X_train, y_train, X_val, y_val


def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    stage_b: dict,
    stage_c: dict,
    n_jobs: int,
    logger: logging.Logger,
):
    def objective(trial: Trial) -> float:
        trial_params = {
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            learning_rate=learning_rate,
            n_estimators=int(stage_b["n_estimators"]),
            max_depth=int(stage_c["max_depth"]),
            random_state=1337,
            reg_lambda=trial_params["reg_lambda"],
            reg_alpha=trial_params["reg_alpha"],
            gamma=trial_params["gamma"],
            min_child_weight=trial_params["min_child_weight"],
            subsample=trial_params["subsample"],
            colsample_bytree=trial_params["colsample_bytree"],
            eval_metric="logloss",
            n_jobs=n_jobs,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        val_logloss = float(log_loss(y_val, val_prob, labels=[0, 1]))
        logger.info("Trial %s: %s val_logloss=%.6f", trial.number, trial_params, val_logloss)
        trial.set_user_attr("val_logloss", val_logloss)
        return val_logloss

    return objective


def main(trials: int = DEFAULT_TRIALS) -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Stage D Regularization Optimization Started ===")
    print("[INFO] Loading Stage A/B/C parameters...")
    lr, stage_b, stage_c = _load_stage_params()
    print(f"[INFO] learning_rate={lr:.6f}")
    print(f"[INFO] Stage B params: {stage_b}")
    print(f"[INFO] Stage C params: {stage_c}")

    print(f"[INFO] Loading datasets (horizon={horizon_hours}h)...")
    X_train, y_train, X_val, y_val = _prepare_data(horizon_hours)
    logger.info(
        "Loaded datasets with horizon=%sh (train=%s, val=%s)",
        horizon_hours,
        X_train.shape,
        X_val.shape,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    objective = objective_factory(
        X_train,
        y_train,
        X_val,
        y_val,
        lr,
        stage_b,
        stage_c,
        n_jobs,
        logger,
    )
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=trials, timeout=None)

    best = study.best_trial
    best_params = best.params.copy()
    best_params.update(
        {
            "learning_rate": lr,
            "n_estimators": stage_b["n_estimators"],
            "max_depth": stage_c["max_depth"],
            "val_logloss": best.value,
        }
    )
    print("[INFO] Best parameters:", best_params)
    print(f"[INFO] Best validation logloss: {best.value:.6f}")
    logger.info("Best trial params: %s", best_params)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    logger.info("Saved best Stage D parameters to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage D Optimization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage D regularization optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage D regularization optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)
