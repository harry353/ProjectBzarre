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


def _load(db: Path, table: str) -> pd.DataFrame:
    """Loads a full table from a SQLite database into a pandas DataFrame."""
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    """
    Standardizes timestamps by converting to UTC, flooring to the hour,
    and removing timezone awareness for database compatibility.
    """
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.floor("h").dt.tz_convert(None)


def _merge(split: str, target: str) -> pd.DataFrame:
    """
    Loads features and labels for a specific split (train/validation/test)
    and merges them on a normalized timestamp.
    """
    f = _load(FEATURES_DB, FEATURE_TABLES[split])
    y = _load(LABELS_DB, LABEL_TABLES[split])

    f["timestamp"] = _normalize_timestamp(f["timestamp"])
    y["timestamp"] = _normalize_timestamp(y["timestamp"])

    return f.merge(
        y[["timestamp", target]],
        on="timestamp",
        how="inner",
    )


def _prepare(df: pd.DataFrame, target: str):
    """
    Prepares features (X), labels (y), and timestamps (ts) for model consumption.
    Handles missing target values and fills missing features with 0.0.
    """
    df = df.dropna(subset=[target])

    ts = df["timestamp"].reset_index(drop=True)
    y = df[target].astype(int).to_numpy()

    # Select only numeric features and convert to float32 for XGBoost
    X = (
        df.drop(columns=["timestamp", target])
        .select_dtypes(include=[np.number])
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    return X, y, ts


def main() -> None:
    """
    Main execution loop iterating over prediction horizons to train models
    and export raw probability predictions for all dataset splits.
    """
    for horizon in TARGET_HORIZONS_H:
        target = f"h_{horizon}"

        # Load and align datasets
        train_df = _merge("train", target)
        val_df = _merge("validation", target)
        test_df = _merge("test", target)

        X_train, y_train, ts_train = _prepare(train_df, target)
        X_val, y_val, ts_val = _prepare(val_df, target)
        X_test, y_test, ts_test = _prepare(test_df, target)

        # Retrieve best hyperparameters from tuning stages
        model_dir = MODEL_ROOT / f"h{horizon}"
        summary_path = model_dir / "summary.json"
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
            params = summary["best_params"]

        # Initialize and fit the XGBoost model
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            **params,
        )

        model.fit(X_train, y_train)

        # Extract probability of the positive class
        train_prob = model.predict_proba(X_train)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        # Consolidate results into a single table
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

        # Export raw probabilities to a local SQLite database for each horizon
        prob_db = model_dir / "raw_probabilities.db"
        prob_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(prob_db) as conn:
            out.to_sql("raw_probs", conn, if_exists="replace", index=False)

        print(f"[OK] Raw probabilities saved to {prob_db}")


if __name__ == "__main__":
    main()
