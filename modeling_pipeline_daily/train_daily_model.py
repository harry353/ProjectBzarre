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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from modeling_pipeline_daily.data_utils import (
    compute_sample_weights,
    feature_columns,
    load_split_tables,
    prepare_arrays,
)
from modeling_pipeline_daily.feature_pruning import select_pruned_features
from modeling_pipeline_daily.plotting import (
    plot_calibration_curve,
    plot_feature_importance,
    plot_pr_with_threshold,
)


MERGED_DB = PROJECT_ROOT / "preprocessing_pipeline" / "final" / "all_preprocessed_sources.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_JSON = OUTPUT_DIR / "daily_storm_features.json"
MODEL_JSON = OUTPUT_DIR / "daily_storm_model.json"
DECISION_JSON = OUTPUT_DIR / "daily_storm_decision_policy.json"
METADATA_JSON = OUTPUT_DIR / "daily_storm_model_metadata.json"

MIN_PRUNED_FEATURES = 70
REL_IMPORTANCE_THRESHOLD = 0.2
N_JOBS = 12
TARGET_RECALL = 0.7
CHOSEN_THRESHOLD_OVERRIDE: float | None = None


def _tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_kwargs: dict,
    sample_weight: np.ndarray | None = None,
    n_trials: int = 30,
) -> tuple[dict, float, optuna.Study]:
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
        return average_precision_score(y_val, val_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value, study


def _train_and_evaluate(splits: Dict[str, pd.DataFrame]) -> None:
    feature_cols = feature_columns(splits["train"])

    train_df_full = splits["train"].copy()
    val_df_full = splits["validation"].copy()
    test_df_full = splits["test"].copy()

    train_sample_weights, severity_stats = compute_sample_weights(train_df_full)

    X_train_full, y_train = prepare_arrays(train_df_full, feature_cols)
    X_val_full, y_val = prepare_arrays(val_df_full, feature_cols)
    X_test_full, y_test = prepare_arrays(test_df_full, feature_cols)

    positives = y_train.sum()
    negatives = len(y_train) - positives
    scale_pos_weight = float(negatives / positives) if positives > 0 else 1.0

    base_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=N_JOBS,
        random_state=41,
    )

    best_params, best_val_ap, study = _tune_hyperparameters(
        X_train_full, y_train, X_val_full, y_val, base_kwargs, sample_weight=train_sample_weights
    )

    prelim_model = XGBClassifier(**base_kwargs, **best_params)
    prelim_model.fit(X_train_full, y_train, sample_weight=train_sample_weights)

    feature_cols = select_pruned_features(
        train_df_full[feature_cols],
        feature_cols,
        prelim_model.feature_importances_,
        MIN_PRUNED_FEATURES,
    )
    with open(FEATURES_JSON, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_order": feature_cols,
                "feature_count": len(feature_cols),
                "pruning_target": MIN_PRUNED_FEATURES,
            },
            fp,
            indent=2,
        )
    print(f"[INFO] Feature contract saved to {FEATURES_JSON}")
    print(f"[INFO] Using {len(feature_cols)} features after pruning.")

    train_df = train_df_full[["forecast_date", "storm_present_next_24h"] + feature_cols].copy()
    val_df = val_df_full[["forecast_date", "storm_present_next_24h"] + feature_cols].copy()
    test_df = test_df_full[["forecast_date", "storm_present_next_24h"] + feature_cols].copy()

    X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df["storm_present_next_24h"].to_numpy(dtype=np.int8)
    X_val = val_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_val = val_df["storm_present_next_24h"].to_numpy(dtype=np.int8)
    X_test = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_test = test_df["storm_present_next_24h"].to_numpy(dtype=np.int8)

    final_model = XGBClassifier(**base_kwargs, **best_params)
    final_model.fit(X_train, y_train, sample_weight=train_sample_weights)
    final_model.get_booster().save_model(MODEL_JSON)
    print(f"[INFO] Final model saved to {MODEL_JSON}")

    val_prob = final_model.predict_proba(X_val)[:, 1]
    test_prob = final_model.predict_proba(X_test)[:, 1]

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, val_prob)
    recall_slice = recall_arr[1:] if len(recall_arr) > 1 else []
    threshold_list = thresholds.tolist()

    if CHOSEN_THRESHOLD_OVERRIDE is not None:
        chosen_threshold = float(CHOSEN_THRESHOLD_OVERRIDE)
        chosen_recall = float(
            recall_score(y_val, (val_prob >= chosen_threshold).astype(int), zero_division=0)
        )
        print(
            f"[INFO] Using override threshold {chosen_threshold:.3f} "
            f"(validation recall {chosen_recall:.3f})."
        )
    else:
        candidates = [
            (float(thr), float(rec))
            for thr, rec in zip(threshold_list, recall_slice)
            if rec >= TARGET_RECALL
        ]
        if candidates:
            chosen_threshold, chosen_recall = candidates[-1]
        elif recall_slice:
            idx = int(
                np.argmin(np.abs(np.asarray(recall_slice, dtype=float) - TARGET_RECALL))
            )
            chosen_threshold = float(threshold_list[idx])
            chosen_recall = float(recall_slice[idx])
        else:
            chosen_threshold = float(threshold_list[-1]) if threshold_list else 0.5
            chosen_recall = float(recall_arr[-1]) if len(recall_arr) else 0.0
        print(
            f"[INFO] Selected threshold {chosen_threshold:.3f} to target recall "
            f"{TARGET_RECALL:.2f} (actual recall {chosen_recall:.3f})."
        )

    val_pred = (val_prob >= chosen_threshold).astype(int)
    val_precision = precision_score(y_val, val_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)

    test_pred = (test_prob >= chosen_threshold).astype(int)

    try:
        calibration_info = plot_calibration_curve(
            y_true=y_val,
            y_prob=val_prob,
            threshold=chosen_threshold,
            png_path=OUTPUT_DIR / "daily_calibration_validation.png",
        )
        pr_val_info = plot_pr_with_threshold(
            y_true=y_val,
            y_prob=val_prob,
            threshold=chosen_threshold,
            png_path=OUTPUT_DIR / "daily_pr_validation.png",
            split_name="Validation",
        )
        pr_test_info = plot_pr_with_threshold(
            y_true=y_test,
            y_prob=test_prob,
            threshold=chosen_threshold,
            png_path=OUTPUT_DIR / "daily_pr_test.png",
            split_name="Test",
        )
        plot_feature_importance(
            feature_cols,
            final_model.feature_importances_,
            OUTPUT_DIR / "daily_feature_importance.png",
        )
    except Exception as exc:
        print(f"[WARN] Plotting diagnostics failed: {exc}")
        calibration_info = {}
        pr_val_info = {}
        pr_test_info = {}

    metrics = {
        "scale_pos_weight": scale_pos_weight,
        "best_hyperparameters": best_params,
        "feature_count": len(feature_cols),
        "severity_weighting": severity_stats,
        "validation_average_precision": float(average_precision_score(y_val, val_prob)),
        "best_validation_average_precision": float(best_val_ap),
        "chosen_threshold": chosen_threshold,
        "target_recall": TARGET_RECALL,
        "validation_precision_at_target_recall": float(val_precision),
        "validation_recall_at_target": float(val_recall),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_average_precision": float(average_precision_score(y_test, test_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "calibration": calibration_info,
        "pr_validation": pr_val_info,
        "pr_test": pr_test_info,
    }

    with open(DECISION_JSON, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "positive_class": "storm_present_next_24h = 1",
                "probability_threshold": float(chosen_threshold),
            },
            fp,
            indent=2,
        )
    print(f"[INFO] Decision policy saved to {DECISION_JSON}")

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": best_params,
        "scale_pos_weight": scale_pos_weight,
        "feature_count": len(feature_cols),
        "pruning_target": MIN_PRUNED_FEATURES,
        "metrics": {
            "validation_average_precision": metrics["validation_average_precision"],
            "test_average_precision": metrics["test_average_precision"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
        },
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"[INFO] Model metadata saved to {METADATA_JSON}")

    print("[INFO] -------- Daily Storm Model Summary --------")
    print(f"[INFO] Threshold        : {chosen_threshold:.3f}")
    print(
        f"[INFO] Validation PR    : AP={metrics['validation_average_precision']:.3f} "
        f"Precision={val_precision:.3f} Recall={val_recall:.3f}"
    )
    print(
        f"[INFO] Test Metrics     : AP={metrics['test_average_precision']:.3f} "
        f"ROC-AUC={metrics['test_roc_auc']:.3f} "
        f"Precision={metrics['test_precision']:.3f} "
        f"Recall={metrics['test_recall']:.3f}"
    )
    print("[INFO] -------------------------------------------")


def main() -> None:
    splits = load_split_tables(MERGED_DB)
    _train_and_evaluate(splits)


if __name__ == "__main__":
    main()
