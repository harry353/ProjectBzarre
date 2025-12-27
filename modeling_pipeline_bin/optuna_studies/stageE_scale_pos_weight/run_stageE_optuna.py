from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Tuple

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
BEST_REG_PATH = PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "best_regularization.json"

LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageE.log"
BEST_SCALE_PATH = Path(__file__).resolve().parent / "best_scale_pos_weight.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageE_scale_pos_weight"
DEFAULT_TRIALS = 40
DEFAULT_N_JOBS = 12


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageE")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_base_params() -> dict:
    params = {}
    params.update(_load_json(BEST_LR_PATH))
    params.update(_load_json(BEST_ITERS_PATH))
    params.update(_load_json(BEST_TREE_PATH))
    params.update(_load_json(BEST_REG_PATH))
    return params


def _prepare_data(horizon_hours: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = load_split_arrays("train", horizon_hours)
    X_val, y_val = load_split_arrays("validation", horizon_hours)

    train_inf_cols = np.unique(np.where(~np.isfinite(X_train))[1]).tolist()
    val_inf_cols = np.unique(np.where(~np.isfinite(X_val))[1]).tolist()
    if train_inf_cols:
        raise ValueError(f"Infinite values detected in training features at indices: {train_inf_cols}")
    if val_inf_cols:
        raise ValueError(f"Infinite values detected in validation features at indices: {val_inf_cols}")

    return X_train, y_train, X_val, y_val


def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: dict,
    n_jobs: int,
    logger: logging.Logger,
):
    def objective(trial: Trial) -> float:
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 20.0, log=True)

        model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            learning_rate=float(base_params["learning_rate"]),
            n_estimators=int(base_params["n_estimators"]),
            max_depth=int(base_params["max_depth"]),
            min_child_weight=float(base_params["min_child_weight"]),
            gamma=float(base_params["gamma"]),
            subsample=float(base_params["subsample"]),
            colsample_bytree=float(base_params["colsample_bytree"]),
            reg_alpha=float(base_params.get("reg_alpha", 0.0)),
            reg_lambda=float(base_params.get("reg_lambda", 1.0)),
            random_state=1337,
            scale_pos_weight=float(scale_pos_weight),
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
        logger.info("Trial %d: spw=%.4f val_logloss=%.6f", trial.number, scale_pos_weight, val_logloss)
        trial.set_user_attr("val_logloss", val_logloss)
        return val_logloss

    return objective


def main(trials: int = DEFAULT_TRIALS) -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", DEFAULT_HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Stage E Scale Pos Weight Optimization Started ===")

    base_params = _load_base_params()
    print(f"[INFO] Using base parameters from previous stages.")

    print(f"[INFO] Loading datasets (horizon={horizon_hours}h)...")
    X_train, y_train, X_val, y_val = _prepare_data(horizon_hours)
    logger.info(
        "Datasets ready: horizon=%sh train=%s val=%s",
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

    objective = objective_factory(X_train, y_train, X_val, y_val, base_params, n_jobs, logger)
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=trials, timeout=None)

    best = study.best_trial
    best_spw = best.params["scale_pos_weight"]
    best_logloss = best.value

    BEST_SCALE_PATH.write_text(
        json.dumps(
            {
                "scale_pos_weight": float(best_spw),
                "val_logloss": float(best_logloss),
                "trial_number": best.number,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Best scale_pos_weight: {best_spw:.4f}")
    print(f"[INFO] Best validation logloss: {best_logloss:.6f}")
    logger.info("Best trial: spw=%.4f val_logloss=%.6f", best_spw, best_logloss)
    logger.info("Best scale_pos_weight saved to %s", BEST_SCALE_PATH)
    logger.info("=== Stage E Optimization Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage E scale_pos_weight optimization")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of Optuna trials")
    args = parser.parse_args()
    main(trials=args.trials)
