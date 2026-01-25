from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
LABELS_DB = PROJECT_ROOT / "preprocessing_pipeline" / "labels" / "dst_regression" / "dst_regression_labels.db"
DST_DB = PROJECT_ROOT / "preprocessing_pipeline" / "merge_features" / "all_preprocessed_sources.db"

FEATURE_TABLES = {
    "train": "pca_train",
    "validation": "pca_validation",
    "test": "pca_test",
}
LABEL_TABLES = {
    "train": "dst_regression_train",
    "validation": "dst_regression_validation",
    "test": "dst_regression_test",
}
DST_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
    "test": "merged_test",
}

HORIZONS = range(1, 9)
ALPHAS = [0.1, 0.5, 0.9]
MODEL_BASE_DIR = PROJECT_ROOT / "regression_pipeline" / "xgb_quantile_regime_aware_model"

SEED = 42
N_TRIALS = 30
DST_THRESHOLD = -20.0

np.random.seed(SEED)


def load(db: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def prepare(split: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    X = load(FEATURES_DB, FEATURE_TABLES[split])
    y = load(LABELS_DB, LABEL_TABLES[split])[["timestamp", target]]
    dst = load(DST_DB, DST_TABLES[split])[["timestamp", "dst_dst"]]

    X["timestamp"] = normalize_ts(X["timestamp"])
    y["timestamp"] = normalize_ts(y["timestamp"])
    dst["timestamp"] = normalize_ts(dst["timestamp"])

    df = X.merge(dst, on="timestamp", how="inner").merge(y, on="timestamp", how="inner")
    df = df[df["dst_dst"] < DST_THRESHOLD].dropna()
    if df.empty:
        raise RuntimeError(f"No rows left for split '{split}' after dst_dst threshold {DST_THRESHOLD}.")

    feature_cols = [
        c for c in df.columns
        if c != target and np.issubdtype(df[c].dtype, np.number)
    ]

    return (
        df[feature_cols].to_numpy(np.float32),
        df[target].to_numpy(np.float32),
    )


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state": SEED,
        "n_jobs": -1,
    }


def objective(trial: optuna.Trial, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, alpha: float) -> float:
    params = suggest_params(trial)
    params.update({"objective": "reg:quantileerror", "quantile_alpha": alpha})

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, verbose=False)

    y_pred = model.predict(X_va)
    return pinball_loss(y_va, y_pred, alpha)


def run_for_alpha(h: int, alpha: float) -> dict:
    target = f"h{h}"

    X_tr, y_tr = prepare("train", target)
    X_va, y_va = prepare("validation", target)
    X_te, y_te = prepare("test", target)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X_tr, y_tr, X_va, y_va, alpha), n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({"random_state": SEED, "n_jobs": -1, "objective": "reg:quantileerror", "quantile_alpha": alpha})

    # Train on train+validation for final model
    X_final = np.vstack([X_tr, X_va])
    y_final = np.concatenate([y_tr, y_va])

    model = XGBRegressor(**best)
    model.fit(X_final, y_final, verbose=False)

    y_pred_tr = model.predict(X_tr)
    y_pred_va = model.predict(X_va)
    y_pred_te = model.predict(X_te)

    metrics = {
        "train": pinball_loss(y_tr, y_pred_tr, alpha),
        "validation": pinball_loss(y_va, y_pred_va, alpha),
        "test": pinball_loss(y_te, y_pred_te, alpha),
    }

    out_dir = MODEL_BASE_DIR / f"h{h}_storm" / f"q{alpha}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"horizon": h, "alpha": alpha, "metrics": metrics}, f, indent=2)

    print(f"\nH{h:02d} | alpha={alpha}")
    print("-" * 44)
    for split in ("train", "validation", "test"):
        print(f"{split:<12}pinball: {metrics[split]:.4f}")

    return metrics


def main() -> None:
    for h in HORIZONS:
        print(f"\n=== Training horizon h{h} for alphas {ALPHAS} ===")
        for alpha in ALPHAS:
            run_for_alpha(h, alpha)


if __name__ == "__main__":
    main()
