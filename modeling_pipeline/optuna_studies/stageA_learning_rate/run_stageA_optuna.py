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
LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageA.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_lr.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageA_learning_rate"


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageA")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def _load_parquet_files(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")
    frames = []
    for file in sorted(directory.glob("*.parquet")):
        frames.append(pd.read_parquet(file))
    if not frames:
        raise ValueError(f"No parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def _prepare_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_df = _load_parquet_files(TRAIN_DIR)
    val_df = _load_parquet_files(VAL_DIR)

    target_cols = [col for col in train_df.columns if col.startswith("target_dst_h6")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected a single target column, found {target_cols}")
    target_col = target_cols[0]

    feature_cols = [col for col in train_df.columns if col != target_col]

    train_nan_cols = train_df.columns[train_df.isna().any()].tolist()
    val_nan_cols = val_df.columns[val_df.isna().any()].tolist()
    if train_nan_cols:
        raise ValueError(f"NaNs detected in training dataset columns: {train_nan_cols}")
    if val_nan_cols:
        raise ValueError(f"NaNs detected in validation dataset columns: {val_nan_cols}")

    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])

    if train_df[feature_cols].isna().any().any():
        raise ValueError("Train features contain NaNs after fill.")
    if val_df[feature_cols].isna().any().any():
        raise ValueError("Validation features contain NaNs after fill.")

    train_inf_mask = ~np.isfinite(train_df[feature_cols])
    val_inf_mask = ~np.isfinite(val_df[feature_cols])
    train_inf_cols = train_inf_mask.columns[train_inf_mask.any()].tolist()
    val_inf_cols = val_inf_mask.columns[val_inf_mask.any()].tolist()
    if train_inf_cols:
        raise ValueError(f"Infinite values detected in training dataset columns: {train_inf_cols}")
    if val_inf_cols:
        raise ValueError(f"Infinite values detected in validation dataset columns: {val_inf_cols}")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)

    return X_train, y_train, X_val, y_val


def objective_factory(X_train, y_train, X_val, y_val, logger: logging.Logger):
    def objective(trial: Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.20, log=True)
        model = XGBRegressor(
            objective="reg:squarederror",
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
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mae = np.mean(np.abs(y_val - val_pred))
        trial.set_user_attr("val_mae", float(val_mae))
        logger.info("Trial %s: lr=%.6f val_mae=%.6f", trial.number, learning_rate, val_mae)
        return val_mae

    return objective


def main() -> None:
    logger = _setup_logger()
    logger.info("=== Stage A Learning Rate Optimization Started ===")
    print("[INFO] Loading datasets...")
    X_train, y_train, X_val, y_val = _prepare_datasets()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    objective = objective_factory(X_train, y_train, X_val, y_val, logger)
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=50, timeout=None)

    best_trial = study.best_trial
    best_lr = best_trial.params["learning_rate"]
    best_mae = best_trial.value

    print(f"Best learning_rate: {best_lr:.6f}")
    print(f"Best validation MAE: {best_mae:.6f}")
    logger.info("Best trial: lr=%.6f val_mae=%.6f", best_lr, best_mae)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "learning_rate": best_lr,
                "val_mae": best_mae,
                "trial_number": best_trial.number,
            },
            fp,
            indent=2,
        )
    logger.info("Best learning rate saved to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage A Optimization Complete ===")


if __name__ == "__main__":
    main()
