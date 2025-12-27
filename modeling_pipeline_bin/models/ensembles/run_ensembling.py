from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from modeling_pipeline_bin.utils.data_loading_bin import (
    DEFAULT_HORIZON_HOURS,
    load_split_dataframe,
)

PRUNED_FEATURES_PATH = PIPELINE_ROOT / "feature_pruning" / "pruned_features.json"
STAGE_A_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
STAGE_B_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
STAGE_C_PATH = PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "best_tree_params.json"
STAGE_D_PATH = PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "best_regularization.json"
STAGE_E_PATH = PIPELINE_ROOT / "optuna_studies" / "stageE_scale_pos_weight" / "best_scale_pos_weight.json"
ENSEMBLE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = ENSEMBLE_DIR / "ensemble_summary.json"
LOG_PATH = PIPELINE_ROOT / "logs" / "training" / "ensemble_training.log"
SEEDS = [1337, 2027, 3037, 4047, 5057]
HORIZON_HOURS = DEFAULT_HORIZON_HOURS
DEFAULT_N_JOBS = 12


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ensemble_training")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing parameter file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_hyperparams() -> dict:
    stage_a = _load_json(STAGE_A_PATH)
    stage_b = _load_json(STAGE_B_PATH)
    stage_c = _load_json(STAGE_C_PATH)
    stage_d = _load_json(STAGE_D_PATH)
    stage_e = _load_json(STAGE_E_PATH)
    params = {
        "learning_rate": stage_a["learning_rate"],
        "n_estimators": int(stage_b["n_estimators"]),
        "early_stopping_rounds": int(stage_b["early_stopping_rounds"]),
        "max_depth": int(stage_d.get("max_depth", stage_c["max_depth"])),
        "min_child_weight": float(stage_d.get("min_child_weight", stage_c["min_child_weight"])),
        "gamma": float(stage_d.get("gamma", stage_c["gamma"])),
        "subsample": float(stage_d.get("subsample", stage_c["subsample"])),
        "colsample_bytree": float(stage_d.get("colsample_bytree", stage_c["colsample_bytree"])),
        "reg_alpha": float(stage_d.get("reg_alpha") or stage_d.get("alpha", 0.0)),
        "reg_lambda": float(stage_d.get("reg_lambda") or stage_d.get("lambda", 1.0)),
        "scale_pos_weight": float(stage_e.get("scale_pos_weight", 1.0)),
    }
    return params


def _load_pruned_features() -> List[str]:
    if not PRUNED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing pruned feature list: {PRUNED_FEATURES_PATH}")
    with PRUNED_FEATURES_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _prepare_data(horizon_hours: int, feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df, y_train = load_split_dataframe("train", horizon_hours)
    val_df, y_val = load_split_dataframe("validation", horizon_hours)
    missing = [feat for feat in feature_names if feat not in train_df.columns]
    if missing:
        raise ValueError(f"Pruned feature list contains columns not present in train set: {missing}")

    feature_cols = feature_names
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_arr = y_train.to_numpy(dtype=np.int8)
    y_val_arr = y_val.to_numpy(dtype=np.int8)

    train_inf_cols = np.unique(np.where(~np.isfinite(X_train))[1]).tolist()
    val_inf_cols = np.unique(np.where(~np.isfinite(X_val))[1]).tolist()
    if train_inf_cols:
        raise ValueError(f"Infinite values detected in training features at indices: {train_inf_cols}")
    if val_inf_cols:
        raise ValueError(f"Infinite values detected in validation features at indices: {val_inf_cols}")

    return X_train, y_train_arr, X_val, y_val_arr


def _build_model(params: dict, seed: int, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        random_state=seed,
        eval_metric="logloss",
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        n_jobs=n_jobs,
    )


def main() -> None:
    horizon_hours = int(os.environ.get("PIPELINE_HORIZON", HORIZON_HOURS))
    n_jobs = int(os.environ.get("PIPELINE_N_JOBS", DEFAULT_N_JOBS))
    logger = _setup_logger()
    logger.info("=== Ensemble training started ===")
    print("[INFO] Loading parameters and pruned features...")
    params = _load_hyperparams()
    pruned_features = _load_pruned_features()
    print(f"[INFO] Pruned feature count: {len(pruned_features)}")

    print(f"[INFO] Loading data (horizon={horizon_hours}h)...")
    X_train, y_train, X_val, y_val = _prepare_data(horizon_hours, pruned_features)
    logger.info(
        "Datasets ready: horizon=%sh train=%s val=%s",
        horizon_hours,
        X_train.shape,
        X_val.shape,
    )

    results = []
    predictions = []
    for seed in SEEDS:
        print(f"[INFO] Training model with seed {seed} ...")
        model = _build_model(params, seed, n_jobs)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_prob = model.predict_proba(X_val)[:, 1]
        val_logloss = float(log_loss(y_val, val_prob, labels=[0, 1]))
        best_iteration = getattr(model, "best_iteration", model.n_estimators)
        print(f"[INFO] Seed {seed}: val_logloss={val_logloss:.6f}, best_iteration={best_iteration}")
        logger.info("Seed %d: val_logloss=%.6f best_iteration=%s", seed, val_logloss, best_iteration)

        model_path = ENSEMBLE_DIR / f"ensemble_model_seed_{seed}.json"
        booster = model.get_booster()
        booster.save_model(model_path)

        results.append(
            {
                "seed": seed,
                "val_logloss": val_logloss,
                "best_iteration": best_iteration,
            }
        )
        predictions.append(val_prob)

    ensemble_prob = np.mean(predictions, axis=0)
    ensemble_logloss = float(log_loss(y_val, ensemble_prob, labels=[0, 1]))
    print(f"[INFO] Ensemble logloss: {ensemble_logloss:.6f}")
    logger.info("Ensemble logloss: %.6f", ensemble_logloss)

    best_single = min(results, key=lambda x: x["val_logloss"])
    if ensemble_logloss + 1e-4 < best_single["val_logloss"]:
        chosen = "ensemble"
        chosen_seed = None
    else:
        chosen = "single"
        chosen_seed = best_single["seed"]

    summary = {
        "seeds": SEEDS,
        "models": results,
        "ensemble_logloss": ensemble_logloss,
        "chosen_strategy": chosen,
        "chosen_seed": chosen_seed,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Ensemble summary saved.")
    logger.info("Chosen strategy: %s (seed=%s)", chosen, chosen_seed)
    logger.info("=== Ensemble training complete ===")


if __name__ == "__main__":
    main()
