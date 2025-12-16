from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from xgboost import XGBRegressor


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PIPELINE_ROOT / "data" / "train"
VAL_DIR = PIPELINE_ROOT / "data" / "validation"
BEST_LR_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageB.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_iters.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageB_iterations"


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageB")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_parquet_dir(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise ValueError(f"No parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def _load_learning_rate() -> float:
    if not BEST_LR_PATH.exists():
        raise FileNotFoundError(f"Stage A best_lr.json not found at {BEST_LR_PATH}")
    with BEST_LR_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    lr = data.get("learning_rate")
    if lr is None:
        raise ValueError(f"learning_rate missing in {BEST_LR_PATH}")
    return float(lr)


def _prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = _load_parquet_dir(TRAIN_DIR)
    val_df = _load_parquet_dir(VAL_DIR)

    target_cols = [col for col in train_df.columns if col.startswith("target_dst_h")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected a single target column, found {target_cols}")
    target_col = target_cols[0]
    feature_cols = [col for col in train_df.columns if col != target_col]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().ffill().bfill()

    train_df = clean(train_df)
    val_df = clean(val_df)

    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])

    if train_df[feature_cols].isna().any().any():
        raise ValueError("NaNs remain in training features after fill.")
    if val_df[feature_cols].isna().any().any():
        raise ValueError("NaNs remain in validation features after fill.")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)
    return X_train, y_train, X_val, y_val


def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    logger: logging.Logger,
):
    def objective(trial: Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2500),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 20, 200),
        }

        model = XGBRegressor(
            objective="reg:squarederror",
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
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_pred = model.predict(X_val)
        val_mae = float(np.mean(np.abs(y_val - val_pred)))
        logger.info(
            "Trial %s: n_estimators=%d max_depth=%d min_child_weight=%.3f early_stop=%d val_mae=%.6f",
            trial.number,
            params["n_estimators"],
            params["max_depth"],
            params["min_child_weight"],
            params["early_stopping_rounds"],
            val_mae,
        )
        trial.set_user_attr("val_mae", val_mae)
        return val_mae

    return objective


def main() -> None:
    logger = _setup_logger()
    logger.info("=== Stage B Iteration Optimization Started ===")
    print("[INFO] Loading Stage A learning rate...")
    lr = _load_learning_rate()
    print(f"[INFO] Using learning_rate={lr:.6f}")

    print("[INFO] Loading datasets...")
    X_train, y_train, X_val, y_val = _prepare_data()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    objective = objective_factory(X_train, y_train, X_val, y_val, lr, logger)
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=60, timeout=None)

    best = study.best_trial
    best_params = best.params.copy()
    best_params["learning_rate"] = lr
    best_params["val_mae"] = best.value
    print("[INFO] Best parameters:", best_params)
    print(f"[INFO] Best validation MAE: {best.value:.6f}")
    logger.info("Best trial params: %s", best_params)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    logger.info("Saved best Stage B parameters to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage B Optimization Complete ===")


if __name__ == "__main__":
    main()
