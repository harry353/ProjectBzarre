from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PIPELINE_ROOT / "data" / "train"
VAL_DIR = PIPELINE_ROOT / "data" / "validation"
TEST_DIR = PIPELINE_ROOT / "data" / "test"
PRUNED_FEATURES_PATH = PIPELINE_ROOT / "feature_pruning" / "pruned_features.json"
STAGE_A_PATH = PIPELINE_ROOT / "optuna_studies" / "stageA_learning_rate" / "best_lr.json"
STAGE_B_PATH = PIPELINE_ROOT / "optuna_studies" / "stageB_iterations" / "best_iters.json"
STAGE_C_PATH = PIPELINE_ROOT / "optuna_studies" / "stageC_tree_params" / "best_tree_params.json"
STAGE_D_PATH = PIPELINE_ROOT / "optuna_studies" / "stageD_regularization" / "best_regularization.json"
ENSEMBLE_SUMMARY_PATH = PIPELINE_ROOT / "models" / "ensembles" / "ensemble_summary.json"
FINAL_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = FINAL_DIR / "final_model_single.json"
FINAL_MODEL_SEED_PATH = FINAL_DIR / "final_model_seed_{seed}.json"
FINAL_ENSEMBLE_INFO = FINAL_DIR / "final_ensemble_info.json"
FINAL_METRICS_PATH = FINAL_DIR / "final_test_metrics.json"
FINAL_PREDICTIONS_PATH = FINAL_DIR / "final_test_predictions.parquet"
LOG_PATH = PIPELINE_ROOT / "logs" / "training" / "final_training.log"


def _setup_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("final_training")
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


def _load_feature_list() -> List[str]:
    return _load_json(PRUNED_FEATURES_PATH)


def _load_hyperparams() -> dict:
    stage_a = _load_json(STAGE_A_PATH)
    stage_b = _load_json(STAGE_B_PATH)
    stage_c = _load_json(STAGE_C_PATH)
    stage_d = _load_json(STAGE_D_PATH)
    params = {
        "learning_rate": float(stage_a["learning_rate"]),
        "n_estimators": int(stage_b["n_estimators"]),
        "max_depth": int(stage_d.get("max_depth", stage_c["max_depth"])),
        "min_child_weight": float(stage_d.get("min_child_weight", stage_c["min_child_weight"])),
        "gamma": float(stage_d.get("gamma", stage_c["gamma"])),
        "subsample": float(stage_d.get("subsample", stage_c["subsample"])),
        "colsample_bytree": float(stage_d.get("colsample_bytree", stage_c["colsample_bytree"])),
        "reg_alpha": float(stage_d.get("reg_alpha") or stage_d.get("alpha", 0.0)),
        "reg_lambda": float(stage_d.get("reg_lambda") or stage_d.get("lambda", 1.0)),
    }
    return params


def _load_dataset(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise ValueError(f"No parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def _prepare_data(feature_list: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = _load_dataset(TRAIN_DIR)
    val_df = _load_dataset(VAL_DIR)
    test_df = _load_dataset(TEST_DIR)

    target_cols = [col for col in train_df.columns if col.startswith("target_dst_h")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected single target column, found {target_cols}")
    target_col = target_cols[0]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().ffill().bfill()

    train_df = clean(train_df)
    val_df = clean(val_df)
    test_df = clean(test_df)

    train_df = train_df.dropna(subset=[target_col])
    val_df = val_df.dropna(subset=[target_col])
    test_df = test_df.dropna(subset=[target_col])

    features = [col for col in feature_list if col in train_df.columns]
    missing = set(feature_list) - set(features)
    if missing:
        raise ValueError(f"Pruned feature list contains missing columns: {missing}")

    def extract(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if df[features].isna().any().any():
            raise ValueError("NaNs remain in feature columns after fill.")
        return df[features].to_numpy(dtype=np.float32), df[target_col].to_numpy(dtype=np.float32)

    X_train, y_train = extract(train_df)
    X_val, y_val = extract(val_df)
    X_test, y_test = extract(test_df)

    X_train_final = np.concatenate([X_train, X_val], axis=0)
    y_train_final = np.concatenate([y_train, y_val], axis=0)

    return X_train_final, y_train_final, X_test, y_test, X_val, y_val


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


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "max_residual": float(residuals.max()),
        "min_residual": float(residuals.min()),
    }
    worst_idx = int(np.argmax(np.abs(residuals)))
    metrics["worst_error"] = float(residuals[worst_idx])
    metrics["worst_index"] = worst_idx
    return metrics, residuals


def main() -> None:
    logger = _setup_logger()
    logger.info("=== Final training started ===")
    print("[INFO] Loading configuration files...")
    params = _load_hyperparams()
    pruned_features = _load_feature_list()
    summary = _load_json(ENSEMBLE_SUMMARY_PATH)
    print(f"[INFO] Chosen strategy: {summary['chosen_strategy']}")

    print("[INFO] Preparing datasets...")
    X_train_final, y_train_final, X_test, y_test, _, _ = _prepare_data(pruned_features)
    logger.info("Train+val shape: %s, Test shape: %s", X_train_final.shape, X_test.shape)

    strategy = summary["chosen_strategy"]
    seeds = summary["seeds"]
    individual_models = summary["models"]

    if strategy == "single":
        chosen_seed = summary["chosen_seed"]
        print(f"[INFO] Training single final model with seed {chosen_seed}")
        model = _build_model(params, chosen_seed)
        model.fit(X_train_final, y_train_final, eval_set=[(X_train_final, y_train_final)], verbose=False)
        y_pred = model.predict(X_test)
        metrics, residuals = _evaluate_predictions(y_test, y_pred)
        model.get_booster().save_model(FINAL_MODEL_PATH)
        ensemble_info = {
            "strategy": "single",
            "seed": chosen_seed,
        }
        FINAL_ENSEMBLE_INFO.write_text(json.dumps(ensemble_info, indent=2), encoding="utf-8")
    elif strategy == "ensemble":
        print("[INFO] Training ensemble of models...")
        preds = []
        seeded_paths = []
        for seed in seeds:
            model = _build_model(params, seed)
            model.fit(X_train_final, y_train_final, eval_set=[(X_train_final, y_train_final)], verbose=False)
            pred = model.predict(X_test)
            preds.append(pred)
            path = FINAL_DIR / f"final_model_seed_{seed}.json"
            model.get_booster().save_model(path)
            seeded_paths.append(str(path))
        y_pred = np.mean(preds, axis=0)
        metrics, residuals = _evaluate_predictions(y_test, y_pred)
        ensemble_info = {
            "strategy": "ensemble",
            "seeds": seeds,
            "model_paths": seeded_paths,
        }
        FINAL_ENSEMBLE_INFO.write_text(json.dumps(ensemble_info, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    FINAL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "pred": y_pred,
            "actual": y_test,
            "residual": residuals,
        }
    ).to_parquet(FINAL_PREDICTIONS_PATH, index=False)

    print("[INFO] Final evaluation metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")
    logger.info("Final metrics: %s", metrics)
    logger.info("Predictions saved to %s", FINAL_PREDICTIONS_PATH)
    logger.info("=== Final training complete ===")


if __name__ == "__main__":
    main()
