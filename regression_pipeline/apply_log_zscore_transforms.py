from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/merge_features/all_preprocessed_sources.db"
)
DEFAULT_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
    "test": "merged_test",
}
DEFAULT_CANDIDATES = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/log_candidates.csv"
)
DEFAULT_OUT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/transformed_features.db"
)

EXCLUDE_SUFFIXES = ("_flag", "_bucket", "_regime", "_source_id")
TIME_COLUMNS = {"timestamp", "time_tag", "date"}


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _load_candidates(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"feature", "log_possible", "has_negatives"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"log_candidates.csv missing required columns: {sorted(missing)}")
    return df


def _is_excluded(name: str) -> bool:
    lowered = name.lower()
    return lowered in TIME_COLUMNS or lowered.endswith(EXCLUDE_SUFFIXES)


def _zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply log->z or z-score-only transforms using log_candidates.csv."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Input SQLite DB.")
    parser.add_argument(
        "--train-table",
        type=str,
        default=DEFAULT_TABLES["train"],
        help="Training table name.",
    )
    parser.add_argument(
        "--validation-table",
        type=str,
        default=DEFAULT_TABLES["validation"],
        help="Validation table name.",
    )
    parser.add_argument(
        "--test-table",
        type=str,
        default=DEFAULT_TABLES["test"],
        help="Test table name.",
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES, help="CSV decisions.")
    parser.add_argument("--out-db", type=Path, default=DEFAULT_OUT_DB, help="Output SQLite DB.")
    parser.add_argument("--out-table-prefix", type=str, default="transformed", help="Output table prefix.")
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"Input DB not found: {args.db}")
    if not args.candidates.exists():
        raise FileNotFoundError(f"log_candidates.csv not found: {args.candidates}")

    tables = {
        "train": args.train_table,
        "validation": args.validation_table,
        "test": args.test_table,
    }
    data = {split: _load_table(args.db, table) for split, table in tables.items()}
    for split, df in data.items():
        if df.empty:
            raise RuntimeError(f"Input table '{tables[split]}' is empty.")

    candidates = _load_candidates(args.candidates)
    candidates = candidates.set_index("feature")

    for split, df in data.items():
        missing = sorted(set(candidates.index) - set(df.columns))
        if missing:
            raise RuntimeError(
                f"Features in log_candidates.csv missing from table '{tables[split]}': {missing}"
            )

    log_z_cols: list[str] = []
    z_only_cols: list[str] = []
    untouched_cols: list[str] = []

    train_df = data["train"]
    for col in train_df.columns:
        if _is_excluded(col):
            untouched_cols.append(col)
            continue

        series = train_df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            untouched_cols.append(col)
            continue

        if col in candidates.index:
            row = candidates.loc[col]
            log_possible = bool(row["log_possible"])
            has_negatives = bool(row["has_negatives"])
        else:
            log_possible = False
            has_negatives = False

        if log_possible:
            for split, df in data.items():
                if (df[col] < 0).any():
                    raise RuntimeError(
                        f"Column '{col}' has negative values in '{tables[split]}'; log transform not allowed."
                    )
            if has_negatives:
                raise RuntimeError(f"Column '{col}' marked has_negatives=True in CSV but log_possible=True.")
            for split in data:
                data[split][col] = np.log1p(data[split][col].astype(float))
            log_z_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            z_only_cols.append(col)
        else:
            untouched_cols.append(col)

    log_z_stats = {}
    for col in log_z_cols:
        train_vals = data["train"][col].astype(float)
        log_z_stats[col] = (float(train_vals.mean()), float(train_vals.std(ddof=0)))

    z_only_stats = {}
    for col in z_only_cols:
        train_vals = data["train"][col].astype(float)
        z_only_stats[col] = (float(train_vals.mean()), float(train_vals.std(ddof=0)))

    for split, df in data.items():
        for col, (mean, std) in log_z_stats.items():
            df[col] = _zscore(df[col].astype(float), mean, std)
        for col, (mean, std) in z_only_stats.items():
            df[col] = _zscore(df[col].astype(float), mean, std)

    args.out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.out_db) as conn:
        for split, df in data.items():
            out_table = f"{args.out_table_prefix}_{split}"
            df.to_sql(out_table, conn, if_exists="replace", index=False)

    print("[OK] Transform complete.")
    print(f"Output: {args.out_db}")
    for split in data:
        print(f"  {args.out_table_prefix}_{split}")
    print(f"Log -> Z features: {len(log_z_cols)}")
    print(f"Z-only features: {len(z_only_cols)}")
    print(f"Untouched features: {len(untouched_cols)}")
    if log_z_cols:
        print("  " + ", ".join(log_z_cols))
    if z_only_cols:
        print("  " + ", ".join(z_only_cols))
    if untouched_cols:
        print("  " + ", ".join(untouched_cols))


if __name__ == "__main__":
    main()
