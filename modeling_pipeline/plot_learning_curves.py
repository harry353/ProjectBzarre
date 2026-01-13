from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, learning_curve
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling_pipeline.data_utils import feature_columns, load_split_tables


MERGED_DB = Path(__file__).resolve().parents[1] / "preprocessing_pipeline" / "merge_features_labels" / "features_with_labels.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "learning_curve.png"
TARGET_HORIZON_H = 1
LABEL_SOURCE = "main_phase"

N_JOBS = 16
RANDOM_STATE = 41
TRAIN_SIZES = np.linspace(0.1, 1.0, 8)
CV_SPLITS = 5


def main() -> None:
    splits = load_split_tables(MERGED_DB, TARGET_HORIZON_H, label_source=LABEL_SOURCE)
    train_df = splits["train"].copy()
    target_col = f"storm_present_next_{TARGET_HORIZON_H}h"
    feature_cols = feature_columns(train_df, target_col)

    X = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = train_df[target_col].to_numpy(dtype=np.int8)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2.0,
        gamma=0.0,
        reg_alpha=1e-6,
        reg_lambda=1.0,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=TRAIN_SIZES,
        cv=cv,
        scoring="average_precision",
        n_jobs=N_JOBS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Train AP")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.plot(train_sizes, val_mean, "o-", label="Validation AP")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    ax.set_title("Learning Curve (Average Precision)")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Average Precision")
    ax.grid(True, alpha=0.3)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"[OK] Learning curve saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
