from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PIPELINE_ROOT / "data" / "train"
VAL_DIR = PIPELINE_ROOT / "data" / "validation"
PRUNED_FEATURES_PATH = PIPELINE_ROOT / "feature_pruning" / "pruned_features.json"
STAGE_A_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
STAGE_B_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
STAGE_C_PATH = PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "best_tree_params.json"
STAGE_D_PATH = PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "best_regularization.json"
ENSEMBLE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = ENSEMBLE_DIR / "ensemble_summary.json"
LOG_PATH = PIPELINE_ROOT / "logs" / "training" / "ensemble_training.log"
SEEDS = [1337, 2027, 3037, 4047, 5057]


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
    }
    return params


def _load_pruned_features() -> List[str]:
    if not PRUNED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing pruned feature list: {PRUNED_FEATURES_PATH}")
    with PRUNED_FEATURES_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_parquet_dir(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise ValueError(f"No parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def _prepare_data(feature_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = _load_parquet_dir(TRAIN_DIR)
    val_df = _load_parquet_dir(VAL_DIR)
    target_cols = [col for col in train_df.columns if col.startswith("target_dst_h")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected single target column, found {target_cols}")
    target_col = target_cols[0]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().ffill().bfill()

    train_df = clean(train_df)
    val_df = clean(val_df)

    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])

    feature_cols = [col for col in feature_names if col in train_df.columns]
    if len(feature_cols) != len(feature_names):
        missing = set(feature_names) - set(feature_cols)
        raise ValueError(f"Pruned feature list contains columns not present in data: {missing}")

    if train_df[feature_cols].isna().any().any():
        raise ValueError("NaNs remain in training features after fill.")
    if val_df[feature_cols].isna().any().any():
        raise ValueError("NaNs remain in validation features after fill.")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)
    return X_train, y_train, X_val, y_val


def _build_model(params: dict, seed: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
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
    )


def main() -> None:
    logger = _setup_logger()
    logger.info("=== Ensemble training started ===")
    print("[INFO] Loading parameters and pruned features...")
    params = _load_hyperparams()
    pruned_features = _load_pruned_features()
    print(f"[INFO] Pruned feature count: {len(pruned_features)}")

    print("[INFO] Loading data...")
    X_train, y_train, X_val, y_val = _prepare_data(pruned_features)

    results = []
    predictions = []
    for seed in SEEDS:
        print(f"[INFO] Training model with seed {seed} ...")
        model = _build_model(params, seed)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_pred = model.predict(X_val)
        val_mae = float(mean_absolute_error(y_val, val_pred))
        best_iteration = getattr(model, "best_iteration", model.n_estimators)
        print(f"[INFO] Seed {seed}: val_mae={val_mae:.6f}, best_iteration={best_iteration}")
        logger.info("Seed %d: val_mae=%.6f best_iteration=%s", seed, val_mae, best_iteration)

        model_path = ENSEMBLE_DIR / f"ensemble_model_seed_{seed}.json"
        booster = model.get_booster()
        booster.save_model(model_path)

        results.append(
            {
                "seed": seed,
                "val_mae": val_mae,
                "best_iteration": best_iteration,
            }
        )
        predictions.append(val_pred)

    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_mae = float(mean_absolute_error(y_val, ensemble_pred))
    print(f"[INFO] Ensemble MAE: {ensemble_mae:.6f}")
    logger.info("Ensemble MAE: %.6f", ensemble_mae)

    best_single = min(results, key=lambda x: x["val_mae"])
    if ensemble_mae + 0.002 < best_single["val_mae"]:
        chosen = "ensemble"
        chosen_seed = None
    else:
        chosen = "single"
        chosen_seed = best_single["seed"]

    summary = {
        "seeds": SEEDS,
        "models": results,
        "ensemble_mae": ensemble_mae,
        "chosen_strategy": chosen,
        "chosen_seed": chosen_seed,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Ensemble summary saved.")
    logger.info("Chosen strategy: %s (seed=%s)", chosen, chosen_seed)
    logger.info("=== Ensemble training complete ===")


if __name__ == "__main__":
    main()
