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
LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageB.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_iters.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageB_iterations"
HORIZON_HOURS = DEFAULT_HORIZON_HOURS
DEFAULT_TRIALS = 60
DEFAULT_N_JOBS = 12
DEFAULT_N_JOBS = 12


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageB")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_learning_rate() -> float:
    if not BEST_LR_PATH.exists():
        raise FileNotFoundError(f"Stage A best_lr.json not found at {BEST_LR_PATH}")
    with BEST_LR_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    lr = data.get("learning_rate")
    if lr is None:
        raise ValueError(f"learning_rate missing in {BEST_LR_PATH}")
    return float(lr)


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
    n_jobs: int,
    logger: logging.Logger,
):
    def objective(trial: Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2500),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 20, 200),
        }

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=1337,
            learning_rate=learning_rate,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
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
        logger.info(
            "Trial %s: n_estimators=%d max_depth=%d min_child_weight=%.3f early_stop=%d val_logloss=%.6f",
            trial.number,
            params["n_estimators"],
            params["max_depth"],
            params["min_child_weight"],
            params["early_stopping_rounds"],
            val_logloss,
        )
        trial.set_user_attr("val_logloss", val_logloss)
        return val_logloss

    return objective


def main(trials: int = DEFAULT_TRIALS) -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Stage B Iteration Optimization Started ===")
    print("[INFO] Loading Stage A learning rate...")
    lr = _load_learning_rate()
    print(f"[INFO] Using learning_rate={lr:.6f}")

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

    objective = objective_factory(X_train, y_train, X_val, y_val, lr, n_jobs, logger)
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=trials, timeout=None)

    best = study.best_trial
    best_params = best.params.copy()
    best_params["learning_rate"] = lr
    best_params["val_logloss"] = best.value
    print("[INFO] Best parameters:", best_params)
    print(f"[INFO] Best validation logloss: {best.value:.6f}")
    logger.info("Best trial params: %s", best_params)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    logger.info("Saved best Stage B parameters to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage B Optimization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage B iteration optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage B iteration optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)
