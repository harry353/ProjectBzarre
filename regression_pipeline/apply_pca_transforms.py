from __future__ import annotations
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
import json


# Input DB containing the log/z-score transformed feature tables produced by the previous pipeline step.
DEFAULT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/transformed_features.db"
)
# Table names for each data split inside the input DB.
DEFAULT_TABLES = {
    "train": "transformed_train",
    "validation": "transformed_validation",
    "test": "transformed_test",
}
# Output DB that will store the PCA-projected train/validation/test tables plus metadata.
DEFAULT_OUT_DB = Path(
    "/home/haris/Documents/ProjectBzarre/regression_pipeline/pca_features.db"
)
# Prefix applied to each output table name (e.g. "pca_train", "pca_validation", "pca_test").
DEFAULT_OUT_PREFIX = "pca"
# Serialized sklearn PCA object — saved so inference can apply the identical projection.
PCA_MODEL_PATH = Path("/home/haris/Documents/ProjectBzarre/regression_pipeline/pca_model.joblib")
# Ordered list of feature names that were fed into PCA — must match at inference time.
PCA_COLUMNS_PATH = Path("/home/haris/Documents/ProjectBzarre/regression_pipeline/pca_columns.json")
# Columns that bypass PCA and are concatenated back into the output unchanged.
PCA_PASSTHROUGH_PATH = Path("/home/haris/Documents/ProjectBzarre/regression_pipeline/pca_passthrough_columns.json")
# Fraction of total variance the selected components must collectively explain.
PCA_VARIANCE_THRESHOLD = 0.95

# Column name suffixes that identify categorical/indicator columns excluded from PCA.
EXCLUDE_SUFFIXES = ("_flag", "_bucket", "_regime", "_source_id")
# Exact column names representing time/date fields — always passed through unchanged.
TIME_COLUMNS = {"timestamp", "time_tag", "date"}


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    # Read the full table from SQLite into a DataFrame.
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _is_excluded(name: str) -> bool:
    # Returns True for time columns and categorical indicator columns that should skip PCA.
    lowered = name.lower()
    return lowered in TIME_COLUMNS or lowered.endswith(EXCLUDE_SUFFIXES)


def _select_pca_columns(df: pd.DataFrame) -> list[str]:
    # Collect all numeric columns that are not explicitly excluded from PCA.
    cols = []
    for col in df.columns:
        if _is_excluded(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _select_passthrough_columns(df: pd.DataFrame, pca_cols: list[str]) -> list[str]:
    # Every column NOT selected for PCA (time, categorical, non-numeric) passes through untouched.
    passthrough = []
    for col in df.columns:
        if col in pca_cols:
            continue
        passthrough.append(col)
    return passthrough


def _write_pca_metadata(
    conn: sqlite3.Connection,
    feature_names: list[str],
    pca: PCA,
) -> None:
    # Pivot the component loadings matrix into a long-format table for easy inspection/querying.
    components = pd.DataFrame(
        pca.components_,
        columns=feature_names,
        index=[f"pc_{i + 1}" for i in range(pca.components_.shape[0])],
    )
    components = (
        components.reset_index()
        .rename(columns={"index": "component"})
        .melt(id_vars=["component"], var_name="feature", value_name="weight")
    )
    # Write per-component, per-feature loading weights to the output DB.
    components.to_sql("pca_components", conn, if_exists="replace", index=False)

    # Build a table with per-component and cumulative explained variance for diagnostics.
    explained = pd.DataFrame(
        {
            "component": [f"pc_{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    explained["cumulative_explained_variance"] = explained["explained_variance_ratio"].cumsum()
    explained.to_sql("pca_explained_variance", conn, if_exists="replace", index=False)


def main() -> None:
    # Configuration — all sourced from module-level defaults (no CLI parsing in this script).
    db_path = DEFAULT_DB
    out_db = DEFAULT_OUT_DB
    out_prefix = DEFAULT_OUT_PREFIX
    tables = DEFAULT_TABLES
    n_components = None  # When None, the variance_threshold determines component count.
    variance_threshold = PCA_VARIANCE_THRESHOLD

    # Pre-flight validation before any data is loaded.
    if not db_path.exists():
        raise FileNotFoundError(f"Input DB not found: {db_path}")
    if n_components is not None and n_components <= 0:
        raise ValueError("n_components must be positive when set.")
    if variance_threshold is not None and not (0.0 < variance_threshold <= 1.0):
        raise ValueError("variance_threshold must be in (0, 1].")

    # Load all three splits into memory and verify none are empty.
    data = {split: _load_table(db_path, table) for split, table in tables.items()}
    for split, df in data.items():
        if df.empty:
            raise RuntimeError(f"Input table '{tables[split]}' is empty.")

    # Determine which columns enter PCA and which bypass it entirely.
    pca_cols = _select_pca_columns(data["train"])
    if not pca_cols:
        raise RuntimeError("No PCA-eligible numeric features found.")
    passthrough_cols = _select_passthrough_columns(data["train"], pca_cols)

    # Fit PCA exclusively on training data to prevent leakage from validation/test splits.
    X_train = data["train"][pca_cols].to_numpy(dtype=float)
    if n_components is not None:
        # Fixed component count requested explicitly.
        pca = PCA(n_components=n_components, whiten=False, svd_solver="full")
    else:
        # Let sklearn select the minimum number of components that satisfy the variance threshold.
        pca = PCA(n_components=variance_threshold, whiten=False, svd_solver="full")
    pca.fit(X_train)
    # Persist PCA model and column ordering for inference reuse
    PCA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, PCA_MODEL_PATH)
    PCA_COLUMNS_PATH.write_text(json.dumps(pca_cols), encoding="utf-8")
    PCA_PASSTHROUGH_PATH.write_text(json.dumps(passthrough_cols), encoding="utf-8")

    # Project all splits through the fitted PCA and reassemble with passthrough columns.
    outputs = {}
    for split, df in data.items():
        X = df[pca_cols].to_numpy(dtype=float)
        comps = pca.transform(X)
        # Name the output dimensions pc_1, pc_2, … pc_N.
        comp_cols = [f"pc_{i + 1}" for i in range(comps.shape[1])]
        pca_df = pd.DataFrame(comps, columns=comp_cols, index=df.index)
        # Concatenate passthrough columns (left) with PCA components (right) into the final table.
        out = pd.concat([df[passthrough_cols].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
        outputs[split] = out

    # Write transformed splits and PCA metadata tables to the output DB.
    out_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(out_db) as conn:
        for split, df in outputs.items():
            out_table = f"{out_prefix}_{split}"
            df.to_sql(out_table, conn, if_exists="replace", index=False)
        _write_pca_metadata(conn, pca_cols, pca)

    # Summary report printed to stdout for quick verification.
    print("[OK] PCA complete.")
    print(f"Input features: {len(pca_cols)}")
    print(f"Components retained: {pca.n_components_}")
    print("Explained variance ratio:", np.array2string(pca.explained_variance_ratio_, precision=6))
    print(
        "Cumulative explained variance:",
        np.array2string(np.cumsum(pca.explained_variance_ratio_), precision=6),
    )
    print(f"Output DB: {out_db}")
    for split in outputs:
        print(f"  {out_prefix}_{split}")


if __name__ == "__main__":
    main()
