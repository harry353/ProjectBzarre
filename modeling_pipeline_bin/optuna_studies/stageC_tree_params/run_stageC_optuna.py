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
LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageC.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_tree_params.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageC_tree_params"
HORIZON_HOURS = DEFAULT_HORIZON_HOURS
DEFAULT_TRIALS = 60
DEFAULT_N_JOBS = 12


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageC")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_stage_params() -> Tuple[float, dict]:
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
    required = ["n_estimators", "early_stopping_rounds", "max_depth", "min_child_weight"]
    for key in required:
        if key not in stage_b:
            raise ValueError(f"{key} missing in Stage B output.")
    return float(lr), stage_b


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
    base_params: dict,
    n_jobs: int,
    logger: logging.Logger,
):
    def objective(trial: Trial) -> float:
        trial_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            learning_rate=learning_rate,
            n_estimators=int(base_params["n_estimators"]),
            random_state=1337,
            max_depth=trial_params["max_depth"],
            min_child_weight=trial_params["min_child_weight"],
            gamma=trial_params["gamma"],
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
        logger.info(
            "Trial %s: %s val_logloss=%.6f",
            trial.number,
            trial_params,
            val_logloss,
        )
        trial.set_user_attr("val_logloss", val_logloss)
        return val_logloss

    return objective


def main(trials: int = DEFAULT_TRIALS) -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Stage C Tree Parameter Optimization Started ===")
    print("[INFO] Loading Stage A & B parameters...")
    learning_rate, stage_b_params = _load_stage_params()
    print(f"[INFO] Using learning_rate={learning_rate:.6f}")
    print(f"[INFO] Stage B parameters: {stage_b_params}")

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
        learning_rate,
        stage_b_params,
        n_jobs,
        logger,
    )
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=trials, timeout=None)

    best = study.best_trial
    best_params = best.params.copy()
    best_params.update(
        {
            "learning_rate": learning_rate,
            "n_estimators": stage_b_params["n_estimators"],
            "early_stopping_rounds": stage_b_params["early_stopping_rounds"],
            "val_logloss": best.value,
        }
    )
    print("[INFO] Best parameters:", best_params)
    print(f"[INFO] Best validation logloss: {best.value:.6f}")
    logger.info("Best trial params: %s", best_params)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    logger.info("Saved best Stage C parameters to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage C Optimization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage C tree parameter optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)
