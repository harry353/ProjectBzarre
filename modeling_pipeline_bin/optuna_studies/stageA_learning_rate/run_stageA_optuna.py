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

LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageA.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_lr.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageA_learning_rate"
HORIZON_HOURS = DEFAULT_HORIZON_HOURS
DEFAULT_TRIALS = 50
DEFAULT_N_JOBS = 12


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageA")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def _prepare_datasets(horizon_hours: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = load_split_arrays("train", horizon_hours)
    X_val, y_val = load_split_arrays("validation", horizon_hours)

    train_inf_cols = np.unique(np.where(~np.isfinite(X_train))[1]).tolist()
    val_inf_cols = np.unique(np.where(~np.isfinite(X_val))[1]).tolist()
    if train_inf_cols:
        raise ValueError(f"Infinite values detected in training features at indices: {train_inf_cols}")
    if val_inf_cols:
        raise ValueError(f"Infinite values detected in validation features at indices: {val_inf_cols}")

    return X_train, y_train, X_val, y_val


def objective_factory(X_train, y_train, X_val, y_val, n_jobs: int, logger: logging.Logger):
    def objective(trial: Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.20, log=True)
        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            max_depth=5,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            n_estimators=900,
            random_state=1337,
            learning_rate=learning_rate,
            eval_metric="logloss",
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_logloss = log_loss(y_val, val_prob, labels=[0, 1])
        trial.set_user_attr("val_logloss", float(val_logloss))
        logger.info("Trial %s: lr=%.6f val_logloss=%.6f", trial.number, learning_rate, val_logloss)
        return val_logloss

    return objective


def main(trials: int = DEFAULT_TRIALS) -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Stage A Learning Rate Optimization Started ===")
    print(f"[INFO] Loading datasets (horizon={horizon_hours}h)...")
    X_train, y_train, X_val, y_val = _prepare_datasets(horizon_hours)
    logger.info(
        "Datasets loaded with horizon=%sh (train=%s, val=%s)",
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

    objective = objective_factory(X_train, y_train, X_val, y_val, n_jobs, logger)
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=trials, timeout=None)

    best_trial = study.best_trial
    best_lr = best_trial.params["learning_rate"]
    best_logloss = best_trial.value

    print(f"Best learning_rate: {best_lr:.6f}")
    print(f"Best validation logloss: {best_logloss:.6f}")
    logger.info("Best trial: lr=%.6f val_logloss=%.6f", best_lr, best_logloss)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "learning_rate": best_lr,
                "val_logloss": best_logloss,
                "trial_number": best_trial.number,
            },
            fp,
            indent=2,
        )
    logger.info("Best learning rate saved to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage A Optimization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage A learning rate optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)
