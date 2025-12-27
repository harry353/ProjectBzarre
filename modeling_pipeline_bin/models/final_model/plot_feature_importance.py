from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Sequence
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


PIPELINE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PIPELINE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

TRAIN_DIR = PIPELINE_ROOT / "data" / "train"
VAL_DIR = PIPELINE_ROOT / "data" / "validation"
TEST_DIR = PIPELINE_ROOT / "data" / "test"
PRUNED_FEATURES_PATH = PIPELINE_ROOT / "feature_pruning" / "pruned_features.json"
OUTPUT_ROOT = PIPELINE_ROOT / "models" / "final_model" / "outputs"
DEFAULT_HORIZON = 4
LABEL_PREFIX = "label_not_quiet_h"
FUTURE_NORM_PREFIX = "dst_future_norm_h"
FUTURE_PHYS_PREFIX = "dst_future_physical_h"
BASE_NON_FEATURE_COLS = {"timestamp", "time_tag"}
EXCLUDED_PREFIXES = (LABEL_PREFIX, FUTURE_NORM_PREFIX, FUTURE_PHYS_PREFIX)
from modeling_pipeline_bin.utils.data_loading import load_split_dataframe, load_split_arrays


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


def _prepare_feature_matrix(horizon: int, features: Sequence[str]) -> tuple[np.ndarray, List[str]]:
    test_df, _ = load_split_dataframe("test", horizon)
    usable_features = [
        f
        for f in features
        if f in test_df.columns
        and f not in BASE_NON_FEATURE_COLS
        and not any(f.startswith(prefix) for prefix in EXCLUDED_PREFIXES)
    ]
    if not usable_features:
        raise RuntimeError("No usable features found for test split.")
    X = test_df[usable_features].to_numpy(dtype=np.float32)
    return X, list(usable_features)


def _load_boosters(info: dict, single_model_path: Path, seed_pattern: Path) -> list[xgb.Booster]:
    boosters: list[xgb.Booster] = []
    strategy = info.get("strategy")
    if strategy == "single":
        if not single_model_path.exists():
            raise FileNotFoundError(f"Final single model not found at {single_model_path}")
        boosters.append(xgb.Booster(model_file=str(single_model_path)))
    elif strategy == "ensemble":
        seeds = info.get("seeds", [])
        if not seeds:
            raise ValueError("Ensemble strategy requires 'seeds' in final_ensemble_info.json")
        for seed in seeds:
            model_path = Path(str(seed_pattern).format(seed=seed))
            if not model_path.exists():
                raise FileNotFoundError(f"Ensemble model missing: {model_path}")
            boosters.append(xgb.Booster(model_file=str(model_path)))
    else:
        raise ValueError("Unknown strategy in ensemble info file.")
    return boosters


def _compute_importance(booster: xgb.Booster, X: np.ndarray) -> np.ndarray:
    default_names = [f"f{i}" for i in range(X.shape[1])]
    dtest = xgb.DMatrix(X, feature_names=default_names)
    contrib = booster.predict(dtest, pred_contribs=True)
    contrib = contrib[:, :-1]  # strip bias term
    return np.mean(np.abs(contrib), axis=0)


def main(horizon: int = DEFAULT_HORIZON) -> None:
    h_dir = OUTPUT_ROOT / f"h{horizon}"
    if not h_dir.exists():
        raise FileNotFoundError(f"Horizon directory missing: {h_dir}")

    label_col = f"{LABEL_PREFIX}{horizon}"
    ensemble_info_path = h_dir / "final_ensemble_info.json"
    single_model_path = h_dir / "final_model_single.json"
    seed_pattern = h_dir / "final_model_seed_{seed}.json"
    output_path = h_dir / "final_feature_importance.png"

    features = _load_pruned_features()
    ensemble_info = _load_json(ensemble_info_path)
    X_test, usable_features = _prepare_feature_matrix(horizon, features)
    boosters = _load_boosters(ensemble_info, single_model_path, seed_pattern)

    importances = np.zeros(len(usable_features), dtype=np.float64)
    for booster in boosters:
        importances += _compute_importance(booster, X_test)
    importances /= len(boosters)

    sorted_indices = np.argsort(importances)[::-1]
    sorted_features = [usable_features[i] for i in sorted_indices]
    sorted_importances = importances[sorted_indices]

    top_n = min(30, len(sorted_features))
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features[:top_n][::-1], sorted_importances[:top_n][::-1])
    plt.title("Final Model Feature Importance (Top {})".format(top_n))
    plt.xlabel("Mean |Contribution|")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved feature importance plot to {output_path}")
    print("[INFO] Top 10 features:")
    for name, score in zip(sorted_features[:100], sorted_importances[:100]):
        print(f"  {name}: {score:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot final model feature importance per horizon")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Horizon to visualize")
    args = parser.parse_args()
    main(horizon=args.horizon)
