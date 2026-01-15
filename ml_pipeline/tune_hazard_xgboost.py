from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "check_multicolinearity"
    / "all_preprocessed_sources.db"
)
LABELS_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "labels"
    / "hazard_label"
    / "storm_onset_hazards.db"
)
OUTPUT_ROOT = PROJECT_ROOT / "ml_pipeline" / "horizon_models"

FEATURE_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
}
LABEL_TABLES = {
    "train": "storm_onset_train",
    "validation": "storm_onset_validation",
}

HORIZONS = range(1, 9)
RANDOM_STATE = 41
N_JOBS = 16
N_TRIALS_COARSE = 30


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return ts.dt.tz_localize(None)


def _merge_split(split: str, target_col: str) -> pd.DataFrame:
    features = _load_table(FEATURES_DB, FEATURE_TABLES[split])
    labels = _load_table(LABELS_DB, LABEL_TABLES[split])[[ "timestamp", target_col ]]
    features["timestamp"] = _normalize_timestamp(features["timestamp"])
    labels["timestamp"] = _normalize_timestamp(labels["timestamp"])
    merged = features.merge(labels, on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError(f"No merged rows for split '{split}'.")
    return merged


def _prepare_xy(df: pd.DataFrame, target_col: str):
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")

    drop_patterns = []
    if drop_patterns:
        drop_cols = [
            c for c in numeric.columns
            if any(pat in c for pat in drop_patterns) and c != target_col
        ]
        if drop_cols:
            numeric = numeric.drop(columns=drop_cols)

    y = numeric[target_col].astype(int).to_numpy()
    X = numeric.drop(columns=[target_col])
    
    feature_cols = X.columns.tolist()

    return X.to_numpy(dtype=np.float32), y, feature_cols


def _train_horizon(horizon: int) -> None:
    target_col = f"h_{horizon}"
    train_df = _merge_split("train", target_col)
    val_df = _merge_split("validation", target_col)

    base_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
    )

    X_train, y_train, feature_cols = _prepare_xy(train_df, target_col)
    X_val, y_val, _ = _prepare_xy(val_df, target_col)

    def coarse_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
            "random_state": RANDOM_STATE,
        }
        model = XGBClassifier(**base_kwargs, **params)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, val_prob)

    coarse_study = optuna.create_study(direction="minimize")
    coarse_study.optimize(
        coarse_objective, n_trials=N_TRIALS_COARSE, show_progress_bar=True
    )

    best_params = coarse_study.best_params
    model = XGBClassifier(**base_kwargs, **best_params)
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    val_logloss = log_loss(y_val, val_prob)
    val_ap = average_precision_score(y_val, val_prob)
    bs = brier_score_loss(y_val, val_prob)
    precision, recall, _ = precision_recall_curve(y_val, val_prob)

    output_dir = OUTPUT_ROOT / f"h{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target_horizon_h": int(horizon),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "val_logloss": float(val_logloss),
        "val_average_precision": float(val_ap),
        "val_brier_score": float(bs),
        "best_params": best_params,
    }

    model_path = output_dir / "model.json"
    model.get_booster().save_model(model_path)

    pr_path = output_dir / "precision_recall_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="tab:blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (h{horizon})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    coarse_study.trials_dataframe().to_csv(
        output_dir / "optuna_trials_coarse.csv", index=False
    )

    print("[INFO] -------- Hazard Model Summary --------")
    print(f"[INFO] Horizon (h)            : {horizon}")
    print(f"[INFO] Features used          : {len(feature_cols)}")
    print(f"[INFO] Train rows             : {len(y_train):,}")
    print(f"[INFO] Validation rows        : {len(y_val):,}")
    print(f"[INFO] Positive rate (train)  : {y_train.mean():.4f}")
    print(f"[INFO] Positive rate (val)    : {y_val.mean():.4f}")
    print(f"[INFO] Val log loss           : {val_logloss:.4f}")
    print(f"[INFO] Val average precision  : {val_ap:.4f}")
    print(f"[INFO] Val Brier score        : {bs:.4f}")
    print(f"[INFO] Model saved to         : {model_path}")
    print(f"[INFO] PR curve saved to      : {pr_path}")
    print("[INFO] Best params:")
    for key in sorted(best_params):
        print(f"[INFO]   {key}: {best_params[key]}")
    print("[INFO] --------------------------------------")


def main() -> None:
    for horizon in HORIZONS:
        _train_horizon(horizon)


if __name__ == "__main__":
    main()
