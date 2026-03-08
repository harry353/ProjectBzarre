from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "merge_features"
    / "all_preprocessed_sources.db"
)
LABELS_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "labels"
    / "main_phase_labels.db"
)
MODEL_ROOT = PROJECT_ROOT / "classification_pipeline_" / "horizon_models"

FEATURE_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
    "test": "merged_test",
}
LABEL_TABLES = {
    "train": "storm_onset_train",
    "validation": "storm_onset_validation",
    "test": "storm_onset_test",
}

TARGET_HORIZONS_H = range(1, 9)


# Loads a full table from a SQLite database into a pandas DataFrame
def _load(db: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


# Standardizes timestamps: convert to UTC, floor to hour, and remove TZ awareness
def _normalize_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.floor("h").dt.tz_convert(None)


# Loads features and labels for a split and merges them on normalized timestamps
def _merge(split: str, target: str) -> pd.DataFrame:
    f = _load(FEATURES_DB, FEATURE_TABLES[split])
    y = _load(LABELS_DB, LABEL_TABLES[split])

    f["timestamp"] = _normalize_timestamp(f["timestamp"])
    y["timestamp"] = _normalize_timestamp(y["timestamp"])

    return f.merge(
        y[["timestamp", target]],
        on="timestamp",
        how="inner",
    )


# Prepares X, y, and timestamps for models; handles missing data and numeric selection
def _prepare(df: pd.DataFrame, target: str):
    df = df.dropna(subset=[target])

    ts = df["timestamp"].reset_index(drop=True)
    y = df[target].astype(int).to_numpy()

    # Fill missing features with zero and ensure float32 precision
    X = (
        df.drop(columns=["timestamp", target])
        .select_dtypes(include=[np.number])
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    return X, y, ts


# Main loop to train models across horizons and export split-wise raw probabilities
def main() -> None:
    for horizon in TARGET_HORIZONS_H:
        target = f"h_{horizon}"

        # 1. Dataset Alignment
        train_df = _merge("train", target)
        val_df = _merge("validation", target)
        test_df = _merge("test", target)

        X_train, y_train, ts_train = _prepare(train_df, target)
        X_val, y_val, ts_val = _prepare(val_df, target)
        X_test, y_test, ts_test = _prepare(test_df, target)

        # 2. Model Initialization
        # Retrieve best hyperparams from previous tuning stages stored in JSON
        model_dir = MODEL_ROOT / f"h{horizon}"
        summary_path = model_dir / "summary.json"
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
            params = summary["best_params"]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            **params,
        )

        # 3. Training and Inference
        model.fit(X_train, y_train)

        # Generate class probabilities for all splits
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        # 4. Result Consolidation
        # Union all split predictions into a single tidy dataframe
        out = pd.concat(
            [
                pd.DataFrame(
                    {
                        "timestamp": ts_train,
                        "y_true": y_train,
                        "y_prob": train_prob,
                        "split": "train",
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": ts_val,
                        "y_true": y_val,
                        "y_prob": val_prob,
                        "split": "validation",
                    }
                ),
                pd.DataFrame(
                    {
                        "timestamp": ts_test,
                        "y_true": y_test,
                        "y_prob": test_prob,
                        "split": "test",
                    }
                ),
            ],
            ignore_index=True,
        )

        # 5. Database Export
        # Persist predictions to horizon-specific SQLite databases
        prob_db = model_dir / "raw_probabilities.db"
        prob_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(prob_db) as conn:
            out.to_sql("raw_probs", conn, if_exists="replace", index=False)

        print(f"[OK] Raw probabilities saved to {prob_db}")


if __name__ == "__main__":
    main()
