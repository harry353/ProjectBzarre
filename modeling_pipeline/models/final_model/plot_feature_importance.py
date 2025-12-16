from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PIPELINE_ROOT / "data" / "train"
VAL_DIR = PIPELINE_ROOT / "data" / "validation"
TEST_DIR = PIPELINE_ROOT / "data" / "test"
PRUNED_FEATURES_PATH = PIPELINE_ROOT / "feature_pruning" / "pruned_features.json"
ENSEMBLE_INFO_PATH = PIPELINE_ROOT / "models" / "final_model" / "final_ensemble_info.json"
FINAL_MODEL_PATH = PIPELINE_ROOT / "models" / "final_model" / "final_model_single.json"
FINAL_MODEL_SEED_PATTERN = PIPELINE_ROOT / "models" / "final_model" / "final_model_seed_{}.json"
OUTPUT_PATH = PIPELINE_ROOT / "models" / "final_model" / "final_feature_importance.png"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_pruned_features() -> list[str]:
    return _load_json(PRUNED_FEATURES_PATH)


def _load_dataset(directory: Path) -> pd.DataFrame:
    if not directory.exists():
        raise FileNotFoundError(f"Missing data directory: {directory}")
    frames = [pd.read_parquet(path) for path in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise ValueError(f"No parquet files found in {directory}")
    return pd.concat(frames, ignore_index=True)


def _prepare_feature_matrix(features: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Use the test set for importance visualization to match final evaluation.
    test_df = _load_dataset(TEST_DIR)
    target_cols = [col for col in test_df.columns if col.startswith("target_dst_h")]
    if len(target_cols) != 1:
        raise ValueError(f"Expected single target column, found {target_cols}")
    target_col = target_cols[0]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy().ffill().bfill()

    test_df = clean(test_df)
    test_df = test_df.dropna(subset=[target_col])

    missing = set(features) - set(test_df.columns)
    if missing:
        raise ValueError(f"Pruned feature list contains columns missing in test set: {missing}")

    X = test_df[features].to_numpy(dtype=np.float32)
    y = test_df[target_col].to_numpy(dtype=np.float32)
    return X, y


def _load_boosters(info: dict) -> list[xgb.Booster]:
    boosters: list[xgb.Booster] = []
    strategy = info.get("strategy")
    if strategy == "single":
        if not FINAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Final single model not found at {FINAL_MODEL_PATH}")
        boosters.append(xgb.Booster(model_file=str(FINAL_MODEL_PATH)))
    elif strategy == "ensemble":
        seeds = info.get("seeds", [])
        if not seeds:
            raise ValueError("Ensemble strategy requires 'seeds' in final_ensemble_info.json")
        for seed in seeds:
            model_path = FINAL_MODEL_SEED_PATTERN.with_name(FINAL_MODEL_SEED_PATTERN.name.format(seed=seed))
            if not model_path.exists():
                raise FileNotFoundError(f"Ensemble model missing: {model_path}")
            boosters.append(xgb.Booster(model_file=str(model_path)))
    else:
        raise ValueError(f"Unknown strategy '{strategy}' in {ENSEMBLE_INFO_PATH}")
    return boosters


def _compute_importance(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    default_names = [f"f{i}" for i in range(X.shape[1])]
    dtest = xgb.DMatrix(X, feature_names=default_names)
    contrib = booster.predict(dtest, pred_contribs=True)
    contrib = contrib[:, :-1]  # strip bias term
    return np.mean(np.abs(contrib), axis=0)


def main() -> None:
    features = _load_pruned_features()
    ensemble_info = _load_json(ENSEMBLE_INFO_PATH)
    X_test, _ = _prepare_feature_matrix(features)
    boosters = _load_boosters(ensemble_info)

    importances = np.zeros(len(features), dtype=np.float64)
    for booster in boosters:
        importances += _compute_importance(booster, X_test)
    importances /= len(boosters)

    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    top_n = min(30, len(sorted_features))
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features[:top_n][::-1], sorted_importances[:top_n][::-1])
    plt.title("Final Model Feature Importance (Top {})".format(top_n))
    plt.xlabel("Mean |Contribution|")
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    print(f"[INFO] Saved feature importance plot to {OUTPUT_PATH}")
    print("[INFO] Top 10 features:")
    for name, score in zip(sorted_features[:100], sorted_importances[:100]):
        print(f"  {name}: {score:.6f}")


if __name__ == "__main__":
    main()
