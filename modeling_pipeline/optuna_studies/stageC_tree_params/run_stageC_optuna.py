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
BEST_ITERS_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
LOG_PATH = PIPELINE_ROOT / "logs" / "optuna" / "stageC.log"
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_tree_params.json"
STUDY_DB = Path(__file__).resolve().parent / "study.db"
STUDY_NAME = "stageC_tree_params"


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stageC")
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


def _prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = _load_parquet_dir(TRAIN_DIR)
    val_df = _load_parquet_dir(VAL_DIR)

    target_cols = [col for col in train_df.columns if col.startswith("target_dst_h")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected single target column, found {target_cols}")
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
    base_params: dict,
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

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            learning_rate=learning_rate,
            n_estimators=int(base_params["n_estimators"]),
            random_state=1337,
            max_depth=trial_params["max_depth"],
            min_child_weight=trial_params["min_child_weight"],
            gamma=trial_params["gamma"],
            subsample=trial_params["subsample"],
            colsample_bytree=trial_params["colsample_bytree"],
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
            "Trial %s: %s val_mae=%.6f",
            trial.number,
            trial_params,
            val_mae,
        )
        trial.set_user_attr("val_mae", val_mae)
        return val_mae

    return objective


def main() -> None:
    logger = _setup_logger()
    logger.info("=== Stage C Tree Parameter Optimization Started ===")
    print("[INFO] Loading Stage A & B parameters...")
    learning_rate, stage_b_params = _load_stage_params()
    print(f"[INFO] Using learning_rate={learning_rate:.6f}")
    print(f"[INFO] Stage B parameters: {stage_b_params}")

    print("[INFO] Loading datasets...")
    X_train, y_train, X_val, y_val = _prepare_data()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
    )

    objective = objective_factory(X_train, y_train, X_val, y_val, learning_rate, stage_b_params, logger)
    print("[INFO] Starting Optuna optimization...")
    study.optimize(objective, n_trials=60, timeout=None)

    best = study.best_trial
    best_params = best.params.copy()
    best_params.update(
        {
            "learning_rate": learning_rate,
            "n_estimators": stage_b_params["n_estimators"],
            "early_stopping_rounds": stage_b_params["early_stopping_rounds"],
            "val_mae": best.value,
        }
    )
    print("[INFO] Best parameters:", best_params)
    print(f"[INFO] Best validation MAE: {best.value:.6f}")
    logger.info("Best trial params: %s", best_params)

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)
    logger.info("Saved best Stage C parameters to %s", BEST_PARAMS_PATH)
    logger.info("=== Stage C Optimization Complete ===")


if __name__ == "__main__":
    main()
