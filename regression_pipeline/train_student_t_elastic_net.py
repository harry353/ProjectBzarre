from __future__ import annotations

import sqlite3
from pathlib import Path
import json
import math
import joblib

import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_DB = PROJECT_ROOT / "regression_pipeline" / "pca_features.db"
LABELS_DB = PROJECT_ROOT / "preprocessing_pipeline" / "labels" / "dst_regression" / "dst_regression_labels.db"

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

HORIZONS = range(2, 9)
MODEL_BASE_DIR = PROJECT_ROOT / "regression_pipeline" / "elasticnet_student_t_models"

SEED = 42
NU_GRID = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 7, 10])
N_TRIALS = 30

np.random.seed(SEED)

def load(db, table):
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)

def normalize_ts(s):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def prepare(split, target):
    X = load(FEATURES_DB, FEATURE_TABLES[split])
    y = load(LABELS_DB, LABEL_TABLES[split])[["timestamp", target]]

    X["timestamp"] = normalize_ts(X["timestamp"])
    y["timestamp"] = normalize_ts(y["timestamp"])

    df = X.merge(y, on="timestamp", how="inner").dropna()

    feature_cols = [
        c for c in df.columns
        if c != target and np.issubdtype(df[c].dtype, np.number)
    ]

    return (
        df[feature_cols].to_numpy(np.float32),
        df[target].to_numpy(np.float32),
    )

def student_t_nll(y, mu, sigma, nu):
    z = (y - mu) / sigma
    log_norm = (
        math.lgamma((nu + 1) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * math.log(nu * math.pi)
    )
    return float(
        np.mean(
            -(log_norm - np.log(sigma) - ((nu + 1) / 2) * np.log1p(z * z / nu))
        )
    )

def select_nu(y_val, mu_val, sigma_val):
    best_nu = None
    best_nll = np.inf
    for nu in NU_GRID:
        nll = student_t_nll(y_val, mu_val, sigma_val, nu)
        if nll < best_nll:
            best_nll = nll
            best_nu = nu
    return best_nu

def objective(trial, X_tr, y_tr, X_va, y_va):
    params = {
        "degree": trial.suggest_int("degree", 1, 4),
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
    }

    mu_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=params["degree"], include_bias=False)),
            ("enet", ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], random_state=SEED, warm_start=True)),
        ]
    )
    mu_model.fit(X_tr, y_tr)

    mu_tr = mu_model.predict(X_tr)
    mu_va = mu_model.predict(X_va)

    res2 = (y_tr - mu_tr) ** 2
    sigma_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=params["degree"], include_bias=False)),
            ("enet", ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], random_state=SEED, warm_start=True)),
        ]
    )
    sigma_model.fit(X_tr, np.log(res2 + 1e-6))

    sigma_va = np.sqrt(np.exp(sigma_model.predict(X_va)))

    nu = select_nu(y_va, mu_va, sigma_va)
    return student_t_nll(y_va, mu_va, sigma_va, nu)

def run(h):
    target = f"h{h}"

    X_tr, y_tr = prepare("train", target)
    X_va, y_va = prepare("validation", target)
    X_te, y_te = prepare("test", target)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_va, y_va),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params

    mu_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=best["degree"], include_bias=False)),
            ("enet", ElasticNet(alpha=best["alpha"], l1_ratio=best["l1_ratio"], random_state=SEED)),
        ]
    )
    mu_model.fit(X_tr, y_tr)

    mu_tr = mu_model.predict(X_tr)
    mu_va = mu_model.predict(X_va)
    mu_te = mu_model.predict(X_te)

    res2 = (y_tr - mu_tr) ** 2
    sigma_model = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=best["degree"], include_bias=False)),
            ("enet", ElasticNet(alpha=best["alpha"], l1_ratio=best["l1_ratio"], random_state=SEED)),
        ]
    )
    sigma_model.fit(X_tr, np.log(res2 + 1e-6))

    sigma_tr = np.sqrt(np.exp(sigma_model.predict(X_tr)))
    sigma_va = np.sqrt(np.exp(sigma_model.predict(X_va)))
    sigma_te = np.sqrt(np.exp(sigma_model.predict(X_te)))

    nu = select_nu(y_va, mu_va, sigma_va)

    nll_tr = student_t_nll(y_tr, mu_tr, sigma_tr, nu)
    nll_va = student_t_nll(y_va, mu_va, sigma_va, nu)
    nll_te = student_t_nll(y_te, mu_te, sigma_te, nu)

    mae_tr = np.mean(np.abs(y_tr - mu_tr))
    mae_va = np.mean(np.abs(y_va - mu_va))
    mae_te = np.mean(np.abs(y_te - mu_te))

    print(f"\nH{h:02d} | nu = {nu:>5}")
    print("-" * 44)
    print(f"{'split':<12}{'NLL':>12}{'MAE':>12}")
    print(f"{'train':<12}{nll_tr:12.4f}{mae_tr:12.4f}")
    print(f"{'validation':<12}{nll_va:12.4f}{mae_va:12.4f}")
    print(f"{'test':<12}{nll_te:12.4f}{mae_te:12.4f}")

    out_dir = MODEL_BASE_DIR / f"h{h}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mu_model, out_dir / "mu_model.joblib")
    joblib.dump(sigma_model, out_dir / "sigma_model.joblib")
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    summary = {
        "horizon": h,
        "nu": nu,
        "seed": SEED,
        "metrics": {
            "train": {"nll": nll_tr, "mae": mae_tr},
            "validation": {"nll": nll_va, "mae": mae_va},
            "test": {"nll": nll_te, "mae": mae_te},
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "nu.txt", "w") as f:
        f.write(str(nu))

def main():
    for h in HORIZONS:
        run(h)

if __name__ == "__main__":
    main()
