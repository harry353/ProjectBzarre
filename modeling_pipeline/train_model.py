from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

from modeling_pipeline.data_utils import (
    compute_sample_weights,
    feature_columns,
    load_split_tables,
    prepare_arrays,
)
from modeling_pipeline.plotting import (
    plot_calibration_curve,
    plot_feature_importance,
    plot_pr_with_threshold,
)

MERGED_DB = PROJECT_ROOT / "preprocessing_pipeline" / "merge_features_labels" / "features_with_labels.db"
LABEL_SOURCE = "main_phase"
TARGET_HORIZONS = range(1, 2)

MIN_PRUNED_FEATURES = 20
N_JOBS = 16
FIXED_THRESHOLD = 0.5


def _tune(
    X_train,
    y_train,
    X_val,
    y_val,
    base_kwargs,
    sample_weight,
    n_trials,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
        }
        model = XGBClassifier(**base_kwargs, **params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= FIXED_THRESHOLD).astype(int)
        return recall_score(y_val, val_pred, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _train_and_evaluate(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    target_col: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df_full = splits["train"].copy()
    val_df_full = splits["validation"].copy()
    test_df_full = splits["test"].copy()

    feature_cols = feature_columns(train_df_full, target_col)

    train_sample_weights, severity_stats = compute_sample_weights(train_df_full)

    X_train_full, y_train = prepare_arrays(train_df_full, feature_cols, target_col)
    X_val_full, y_val = prepare_arrays(val_df_full, feature_cols, target_col)
    X_test_full, y_test = prepare_arrays(test_df_full, feature_cols, target_col)

    base_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=41,
    )

    coarse_params, coarse_recall = _tune(
        X_train_full,
        y_train,
        X_val_full,
        y_val,
        base_kwargs,
        train_sample_weights,
        n_trials=15,
    )

    coarse_model = XGBClassifier(**base_kwargs, **coarse_params)
    coarse_model.fit(X_train_full, y_train, sample_weight=train_sample_weights)

    perm = permutation_importance(
        coarse_model,
        X_val_full,
        y_val,
        n_repeats=10,
        random_state=41,
        n_jobs=N_JOBS,
        scoring="recall",
    )

    importances = perm.importances_mean
    order = np.argsort(importances)[::-1]
    selected_idx = order[:MIN_PRUNED_FEATURES]
    feature_cols = [feature_cols[i] for i in selected_idx]

    X_train, y_train = prepare_arrays(train_df_full, feature_cols, target_col)
    X_val, y_val = prepare_arrays(val_df_full, feature_cols, target_col)
    X_test, y_test = prepare_arrays(test_df_full, feature_cols, target_col)

    final_params, final_val_recall = _tune(
        X_train,
        y_train,
        X_val,
        y_val,
        base_kwargs,
        train_sample_weights,
        n_trials=30,
    )

    final_model = XGBClassifier(**base_kwargs, **final_params)
    final_model.fit(X_train, y_train, sample_weight=train_sample_weights)

    model_path = output_dir / "daily_storm_model.json"
    final_model.get_booster().save_model(model_path)

    val_prob = final_model.predict_proba(X_val)[:, 1]
    test_prob = final_model.predict_proba(X_test)[:, 1]

    val_pred = (val_prob >= FIXED_THRESHOLD).astype(int)
    test_pred = (test_prob >= FIXED_THRESHOLD).astype(int)

    plot_feature_importance(
        feature_cols,
        final_model.feature_importances_,
        output_dir / "daily_feature_importance.png",
    )
    plot_calibration_curve(
        y_true=y_val,
        y_prob=val_prob,
        threshold=FIXED_THRESHOLD,
        png_path=output_dir / "daily_calibration_validation.png",
    )
    plot_pr_with_threshold(
        y_true=y_val,
        y_prob=val_prob,
        threshold=FIXED_THRESHOLD,
        png_path=output_dir / "daily_pr_validation.png",
        split_name="Validation",
    )
    plot_pr_with_threshold(
        y_true=y_test,
        y_prob=test_prob,
        threshold=FIXED_THRESHOLD,
        png_path=output_dir / "daily_pr_test.png",
        split_name="Test",
    )

    val_metrics = {
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "accuracy": accuracy_score(y_val, val_pred),
        "average_precision": average_precision_score(y_val, val_prob),
        "roc_auc": roc_auc_score(y_val, val_prob),
    }

    test_metrics = {
        "recall": recall_score(y_test, test_pred, zero_division=0),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, test_pred),
        "average_precision": average_precision_score(y_test, test_prob),
        "roc_auc": roc_auc_score(y_test, test_prob),
    }

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": final_params,
        "feature_count": len(feature_cols),
        "pruning_method": "permutation_importance_validation",
        "severity_weighting": severity_stats,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(output_dir / "daily_storm_model_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print("[INFO] -------- Daily Storm Model Summary --------")
    print(f"[INFO] Features used        : {len(feature_cols)}")
    print(f"[INFO] Fixed threshold      : {FIXED_THRESHOLD:.2f}")
    print(
        f"[INFO] Validation metrics  : "
        f"Recall={val_metrics['recall']:.3f} "
        f"Precision={val_metrics['precision']:.3f} "
        f"AP={val_metrics['average_precision']:.3f} "
        f"ROC-AUC={val_metrics['roc_auc']:.3f}"
    )
    print(
        f"[INFO] Test metrics        : "
        f"Recall={test_metrics['recall']:.3f} "
        f"Precision={test_metrics['precision']:.3f} "
        f"AP={test_metrics['average_precision']:.3f} "
        f"ROC-AUC={test_metrics['roc_auc']:.3f}"
    )
    print("[INFO] ------------------------------------------")


def main() -> None:
    for horizon in TARGET_HORIZONS:
        target_col = f"storm_present_next_{horizon}h"
        output_dir = Path(__file__).resolve().parent / f"output_h{horizon}"
        splits = load_split_tables(MERGED_DB, horizon, label_source=LABEL_SOURCE)
        _train_and_evaluate(splits, output_dir, target_col)


if __name__ == "__main__":
    main()
