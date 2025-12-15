from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DB = PIPELINE_ROOT / "6_normalization" / "space_weather_aver_imp_eng_split_norm.db"
TRAIN_TABLE = "space_weather_train"
VAL_TABLE = "space_weather_validation"
TEST_TABLE = "space_weather_test"
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_COLUMN = "dst"
CATEGORICAL_EXCLUDES = {"cme_last_halo_class"}
META_COLUMNS = {"timestamp"}


def _validate_horizon(horizon: int) -> None:
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Forecast horizon must be a positive integer number of hours.")


def _ensure_column_alignment(df_train: pd.DataFrame, df_other: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [col for col in df_train.columns if col not in df_other.columns]
    extra = [col for col in df_other.columns if col not in df_train.columns]
    if missing or extra:
        raise ValueError(f"{name} columns do not match training columns.")
    return df_other[df_train.columns]


def _select_predictor_columns(columns: Sequence[str]) -> List[str]:
    predictors: List[str] = []
    for col in columns:
        if col == TARGET_COLUMN:
            continue
        if col in CATEGORICAL_EXCLUDES:
            continue
        if col in META_COLUMNS:
            continue
        predictors.append(col)
    return predictors


def _build_split_matrices(
    df: pd.DataFrame,
    predictor_columns: Sequence[str],
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    shifted_target = df[TARGET_COLUMN].shift(-horizon)
    valid_mask = shifted_target.notna()
    X = df.loc[valid_mask, predictor_columns]
    Y = shifted_target.loc[valid_mask]
    X = X.to_numpy(dtype=np.float32, copy=True)
    Y = Y.to_numpy(dtype=np.float32, copy=True)
    return X, Y


def _persist_to_db(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    features: Sequence[str],
    horizon: int,
) -> Path:
    target_name = f"target_dst_h{horizon}"
    output_path = OUTPUT_DIR / f"space_weather_aver_imp_eng_split_norm_h{horizon}.db"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _frame(X: np.ndarray, Y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=features)
        df[target_name] = Y
        return df

    df_train_out = _frame(X_train, Y_train)
    df_val_out = _frame(X_val, Y_val)
    df_test_out = _frame(X_test, Y_test)

    with sqlite3.connect(output_path) as conn:
        df_train_out.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        df_val_out.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        df_test_out.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)
    return output_path


def create_supervised_targets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    _validate_horizon(horizon)

    df_val = _ensure_column_alignment(df_train, df_val, "Validation")
    df_test = _ensure_column_alignment(df_train, df_test, "Test")

    predictor_columns = _select_predictor_columns(df_train.columns)
    if not predictor_columns:
        raise ValueError("No predictor columns remain after applying exclusion rules.")

    X_train, Y_train = _build_split_matrices(df_train, predictor_columns, horizon)
    X_val, Y_val = _build_split_matrices(df_val, predictor_columns, horizon)
    X_test, Y_test = _build_split_matrices(df_test, predictor_columns, horizon)

    num_features = len(predictor_columns)
    print(f"Horizon: {horizon} hour(s)")
    print(f"Number of features: {num_features}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, predictor_columns


if __name__ == "__main__":
    example_horizon = 6
    with sqlite3.connect(NORMALIZED_DB) as conn:
        df_train = pd.read_sql_query(f"SELECT * FROM {TRAIN_TABLE}", conn)
        df_val = pd.read_sql_query(f"SELECT * FROM {VAL_TABLE}", conn)
        df_test = pd.read_sql_query(f"SELECT * FROM {TEST_TABLE}", conn)

    X_train, Y_train, X_val, Y_val, X_test, Y_test, features = create_supervised_targets(
        df_train, df_val, df_test, horizon=example_horizon
    )
    output_db = _persist_to_db(X_train, Y_train, X_val, Y_val, X_test, Y_test, features, example_horizon)
    print(f"Example run complete. Saved matrices to {output_db}")
