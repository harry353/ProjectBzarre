from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
from typing import List, Sequence

import pandas as pd

STAGE_DIR = Path(__file__).resolve().parent
NORMALIZED_DB = STAGE_DIR.parents[1] / "dst" / "7_normalization" / "dst_norm.db"
TRAIN_TABLE = "dst_train"
VAL_TABLE = "dst_validation"
TEST_TABLE = "dst_test"
TARGET_COLUMN = "dst"


def _load_split(table: str) -> pd.DataFrame:
    with sqlite3.connect(NORMALIZED_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    return df


def _select_predictors(columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col not in {"timestamp", TARGET_COLUMN}]


def _align_columns(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in reference.columns if col not in other.columns]
    for col in missing:
        other[col] = 0.0
    extra = [col for col in other.columns if col not in reference.columns]
    if extra:
        other = other.drop(columns=extra)
    return other[reference.columns]


def _build_supervised(df: pd.DataFrame, predictors: Sequence[str], horizon: int) -> pd.DataFrame:
    shifted = df[TARGET_COLUMN].shift(-horizon)
    mask = shifted.notna()
    features = df.loc[mask, predictors].copy()
    features[f"target_{TARGET_COLUMN}_h{horizon}"] = shifted.loc[mask]
    return features


def build_dst_supervised(horizon: int) -> None:
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")

    train = _load_split(TRAIN_TABLE)
    val = _load_split(VAL_TABLE)
    test = _load_split(TEST_TABLE)

    val = _align_columns(train, val)
    test = _align_columns(train, test)

    predictors = _select_predictors(train.columns)
    if not predictors:
        raise RuntimeError("No predictor columns available for supervised dataset.")

    train_sup = _build_supervised(train, predictors, horizon)
    val_sup = _build_supervised(val, predictors, horizon)
    test_sup = _build_supervised(test, predictors, horizon)

    output_db = STAGE_DIR / f"dst_h{horizon}.db"
    with sqlite3.connect(output_db) as conn:
        train_sup.to_sql(TRAIN_TABLE, conn, if_exists="replace", index=False)
        val_sup.to_sql(VAL_TABLE, conn, if_exists="replace", index=False)
        test_sup.to_sql(TEST_TABLE, conn, if_exists="replace", index=False)

    print(f"[OK] Supervised matrices created for horizon h={horizon}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DST supervised training matrices.")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon in hours.")
    args = parser.parse_args()
    build_dst_supervised(horizon=args.horizon)


if __name__ == "__main__":
    main()
