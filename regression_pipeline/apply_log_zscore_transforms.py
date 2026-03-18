from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# Default path to the merged, preprocessed SQLite database produced by the preprocessing pipeline.
DEFAULT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/merge_features/all_preprocessed_sources.db"
)
# Table names for each data split inside the input DB.
DEFAULT_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
    "test": "merged_test",
}
# CSV that documents, for each candidate feature, whether a log transform is safe to apply.
DEFAULT_CANDIDATES = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/log_candidates.csv"
)
# Output SQLite DB that will hold the fully transformed train/validation/test tables.
DEFAULT_OUT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/transformed_features.db"
)

# Column name suffixes that should never be transformed (categorical / indicator columns).
EXCLUDE_SUFFIXES = ("_flag", "_bucket", "_regime", "_source_id")
# Exact column names that represent time/date fields and must pass through untouched.
TIME_COLUMNS = {"timestamp", "time_tag", "date"}


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    # Open a read-only connection and pull the entire table into memory as a DataFrame.
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _load_candidates(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Validate that the CSV contains the three columns this script depends on.
    required = {"feature", "log_possible", "has_negatives"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"log_candidates.csv missing required columns: {sorted(missing)}")
    return df


def _is_excluded(name: str) -> bool:
    # A column should be left untouched if it is a time column or carries a categorical suffix.
    lowered = name.lower()
    return lowered in TIME_COLUMNS or lowered.endswith(EXCLUDE_SUFFIXES)


def _zscore(series: pd.Series, mean: float, std: float) -> pd.Series:
    # Guard against zero or NaN std — return an all-zero series to avoid division by zero.
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

    # Fail fast if required inputs are absent before any expensive work begins.
    if not args.db.exists():
        raise FileNotFoundError(f"Input DB not found: {args.db}")
    if not args.candidates.exists():
        raise FileNotFoundError(f"log_candidates.csv not found: {args.candidates}")

    # Map split names to their respective table names for convenient iteration.
    tables = {
        "train": args.train_table,
        "validation": args.validation_table,
        "test": args.test_table,
    }
    # Load all three splits into memory at once.
    data = {split: _load_table(args.db, table) for split, table in tables.items()}
    # Sanity-check: every split must contain at least one row.
    for split, df in data.items():
        if df.empty:
            raise RuntimeError(f"Input table '{tables[split]}' is empty.")

    # Load the per-feature log-transform decisions and index by feature name for O(1) lookup.
    candidates = _load_candidates(args.candidates)
    candidates = candidates.set_index("feature")

    # Ensure every feature listed in the CSV actually exists in each data split.
    for split, df in data.items():
        missing = sorted(set(candidates.index) - set(df.columns))
        if missing:
            raise RuntimeError(
                f"Features in log_candidates.csv missing from table '{tables[split]}': {missing}"
            )

    # Buckets that track which transform (if any) was applied to each column.
    log_z_cols: list[str] = []   # columns that receive log1p then z-score
    z_only_cols: list[str] = []  # numeric columns that receive only z-score
    untouched_cols: list[str] = []  # columns that pass through unchanged

    # Use the training split to determine column dtypes and candidate decisions.
    train_df = data["train"]
    for col in train_df.columns:
        # Skip time/date and categorical indicator columns entirely.
        if _is_excluded(col):
            untouched_cols.append(col)
            continue

        series = train_df[col]
        # Skip datetime64 columns — they cannot be numerically transformed.
        if pd.api.types.is_datetime64_any_dtype(series):
            untouched_cols.append(col)
            continue

        # Look up the CSV decision for this column; default to no-log if not listed.
        if col in candidates.index:
            row = candidates.loc[col]
            log_possible = bool(row["log_possible"])
            has_negatives = bool(row["has_negatives"])
        else:
            log_possible = False
            has_negatives = False

        if log_possible:
            # Double-check that no split contains actual negatives at runtime,
            # which would make log1p produce NaN/incorrect results.
            for split, df in data.items():
                if (df[col] < 0).any():
                    raise RuntimeError(
                        f"Column '{col}' has negative values in '{tables[split]}'; log transform not allowed."
                    )
            # The CSV flags are mutually exclusive: a column cannot both have negatives and be log-safe.
            if has_negatives:
                raise RuntimeError(f"Column '{col}' marked has_negatives=True in CSV but log_possible=True.")
            # Apply log1p in-place across all three splits before z-scoring.
            for split in data:
                data[split][col] = np.log1p(data[split][col].astype(float))
            log_z_cols.append(col)
            continue

        # Non-log numeric columns still get z-scored for scale consistency.
        if pd.api.types.is_numeric_dtype(series):
            z_only_cols.append(col)
        else:
            untouched_cols.append(col)

    # Compute mean and population std from the training split only (no data leakage).
    log_z_stats = {}
    for col in log_z_cols:
        train_vals = data["train"][col].astype(float)
        log_z_stats[col] = (float(train_vals.mean()), float(train_vals.std(ddof=0)))

    z_only_stats = {}
    for col in z_only_cols:
        train_vals = data["train"][col].astype(float)
        z_only_stats[col] = (float(train_vals.mean()), float(train_vals.std(ddof=0)))

    # Apply the training-derived statistics to all splits (train, validation, test).
    for split, df in data.items():
        for col, (mean, std) in log_z_stats.items():
            df[col] = _zscore(df[col].astype(float), mean, std)
        for col, (mean, std) in z_only_stats.items():
            df[col] = _zscore(df[col].astype(float), mean, std)

    # Write all three transformed splits to the output SQLite DB, replacing any existing tables.
    args.out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.out_db) as conn:
        for split, df in data.items():
            out_table = f"{args.out_table_prefix}_{split}"
            df.to_sql(out_table, conn, if_exists="replace", index=False)

    # Summary report so the caller can confirm what was transformed.
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
