from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
)
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
    / "main_phase_labels.db"
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
N_TRIALS_FINE = 40
MIN_FEATURES_TO_KEEP = 50


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
    labels = _load_table(LABELS_DB, LABEL_TABLES[split])[["timestamp", target_col]]
    features["timestamp"] = _normalize_timestamp(features["timestamp"])
    labels["timestamp"] = _normalize_timestamp(labels["timestamp"])
    merged = features.merge(labels, on="timestamp", how="inner")
    if merged.empty:
        raise RuntimeError(f"No merged rows for split '{split}'.")
    return merged


def _prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
):
    numeric = df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")
    if feature_cols is not None:
        numeric = numeric[[target_col, *feature_cols]]
    y = numeric[target_col].astype(int).to_numpy()
    X = numeric.drop(columns=[target_col])
    return X.to_numpy(dtype=np.float32), y, X.columns.tolist()


def _embedded_prune_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    base_kwargs: dict,
    params: dict,
    min_gain_fraction: float = 0.01,
    min_features: int = MIN_FEATURES_TO_KEEP,
):
    if X.shape[1] <= min_features:
        return X, feature_cols

    model = XGBClassifier(**base_kwargs, **params)
    model.fit(X, y)

    scores = model.get_booster().get_score(importance_type="gain")

    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        total_gain = sum(scores.values())

        keep = [
            f for f, g in ranked
            if g / total_gain >= min_gain_fraction
        ]
    else:
        keep = []

    if len(keep) < min_features:
        keep = feature_cols[:min_features]

    idx = [i for i, f in enumerate(feature_cols) if f in keep]

    if len(idx) < min_features:
        idx = list(range(min_features))

    return X[:, idx], [feature_cols[i] for i in idx]


def _train_horizon(horizon: int) -> None:
    target_col = f"h_{horizon}"

    train_df = _merge_split("train", target_col)
    val_df = _merge_split("validation", target_col)

    X_train, y_train, feature_cols = _prepare_xy(train_df, target_col)
    X_val, y_val, _ = _prepare_xy(val_df, target_col, feature_cols)

    base_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )

    def coarse_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0),
        }
        model = XGBClassifier(**base_kwargs, **params)
        model.fit(X_train, y_train)
        return log_loss(y_val, model.predict_proba(X_val)[:, 1])

    coarse = optuna.create_study(direction="minimize")
    coarse.optimize(coarse_objective, n_trials=N_TRIALS_COARSE, show_progress_bar=True)

    X_train_p, feature_cols_p = _embedded_prune_features(
        X_train,
        y_train,
        feature_cols,
        base_kwargs,
        coarse.best_params,
        min_gain_fraction=0.01,
    )
    assert X_train_p.shape[1] >= MIN_FEATURES_TO_KEEP

    idx = [feature_cols.index(f) for f in feature_cols_p]
    X_val_p = X_val[:, idx]

    def fine_objective(trial: optuna.Trial) -> float:
        bp = coarse.best_params
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                max(50, int(bp["n_estimators"] * 0.7)),
                int(bp["n_estimators"] * 1.3),
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                max(1, bp["max_depth"] - 2),
                bp["max_depth"] + 2,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                bp["learning_rate"] * 0.5,
                bp["learning_rate"] * 1.5,
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                max(0.3, bp["subsample"] - 0.2),
                min(1.0, bp["subsample"] + 0.2),
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                max(0.3, bp["colsample_bytree"] - 0.2),
                min(1.0, bp["colsample_bytree"] + 0.2),
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                max(0.1, bp["min_child_weight"] * 0.5),
                bp["min_child_weight"] * 1.5,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                max(0.0, bp["gamma"] * 0.5),
                bp["gamma"] * 1.5,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                bp["reg_alpha"] * 0.5,
                bp["reg_alpha"] * 2.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                bp["reg_lambda"] * 0.5,
                bp["reg_lambda"] * 2.0,
            ),
        }
        model = XGBClassifier(**base_kwargs, **params)
        model.fit(X_train_p, y_train)
        return log_loss(y_val, model.predict_proba(X_val_p)[:, 1])

    fine = optuna.create_study(direction="minimize")
    fine.optimize(fine_objective, n_trials=N_TRIALS_FINE, show_progress_bar=True)

    model = XGBClassifier(**base_kwargs, **fine.best_params)
    model.fit(X_train_p, y_train)

    val_prob = model.predict_proba(X_val_p)[:, 1]

    out = OUTPUT_ROOT / f"h{horizon}"
    out.mkdir(parents=True, exist_ok=True)

    model.get_booster().save_model(out / "model.json")

    with (out / "selected_features.json").open("w") as f:
        json.dump(feature_cols_p, f, indent=2)

    with (out / "summary.json").open("w") as f:
        json.dump(
            {
                "horizon": horizon,
                "feature_count": len(feature_cols_p),
                "val_logloss": log_loss(y_val, val_prob),
                "val_average_precision": average_precision_score(y_val, val_prob),
                "val_brier_score": brier_score_loss(y_val, val_prob),
                "best_params": fine.best_params,
            },
            f,
            indent=2,
        )

    precision, recall, _ = precision_recall_curve(y_val, val_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.tight_layout()
    plt.savefig(out / "precision_recall_curve.png", dpi=200)
    plt.close()


def main() -> None:
    for h in HORIZONS:
        _train_horizon(h)


if __name__ == "__main__":
    main()
