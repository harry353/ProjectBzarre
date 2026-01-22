from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Inputs/outputs
IN_DB = PROJECT_ROOT / "inference" / "preprocessed_vector_1m.db"
PCA_OUT_DB = PROJECT_ROOT / "inference" / "regression" / "pca_regression_vector_1m.db"
PCA_OUT_TABLE = "pca_inference_vector"

# Paths to training-time assets
LOG_Z_SCRIPT = PROJECT_ROOT / "regression_pipeline" / "apply_log_zscore_transforms.py"
LOG_CANDIDATES = PROJECT_ROOT / "regression_pipeline" / "log_candidates.csv"
PCA_MODEL_PATH = PROJECT_ROOT / "regression_pipeline" / "pca_model.joblib"
PCA_COLUMNS_PATH = PROJECT_ROOT / "regression_pipeline" / "pca_columns.json"
PCA_PASSTHROUGH_PATH = PROJECT_ROOT / "regression_pipeline" / "pca_passthrough_columns.json"


def _run_log_zscore() -> None:
    if not IN_DB.exists():
        raise FileNotFoundError(f"Input DB not found: {IN_DB}")
    if not LOG_CANDIDATES.exists():
        raise FileNotFoundError(f"log_candidates.csv not found: {LOG_CANDIDATES}")
    if not LOG_Z_SCRIPT.exists():
        raise FileNotFoundError(f"Transform script not found: {LOG_Z_SCRIPT}")

    cmd = [
        sys.executable,
        str(LOG_Z_SCRIPT),
        "--db",
        str(IN_DB),
        "--train-table",
        "inference_vector",
        "--validation-table",
        "inference_vector",
        "--test-table",
        "inference_vector",
        "--candidates",
        str(LOG_CANDIDATES),
        "--out-db",
        str(PCA_OUT_DB),
        "--out-table-prefix",
        "transformed_inf",
    ]
    subprocess.run(cmd, check=True)


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _apply_saved_pca() -> None:
    if not PCA_MODEL_PATH.exists() or not PCA_COLUMNS_PATH.exists() or not PCA_PASSTHROUGH_PATH.exists():
        raise FileNotFoundError("PCA artifacts not found. Run apply_pca_transforms.py first.")
    pca = joblib.load(PCA_MODEL_PATH)
    pca_cols = json.loads(PCA_COLUMNS_PATH.read_text(encoding="utf-8"))
    passthrough_cols = json.loads(PCA_PASSTHROUGH_PATH.read_text(encoding="utf-8"))

    # Use the transformed "train" table (all splits identical here).
    table_name = "transformed_inf_train"
    df = _load_table(PCA_OUT_DB, table_name)
    if df.empty:
        raise RuntimeError(f"Input table '{table_name}' is empty.")
    missing = [c for c in pca_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required PCA columns: {missing}")
    missing_pass = [c for c in passthrough_cols if c not in df.columns]
    if missing_pass:
        raise RuntimeError(f"Missing passthrough columns: {missing_pass}")

    X = df[pca_cols].to_numpy(dtype=float)
    comps = pca.transform(X)
    comp_cols = [f"pc_{i + 1}" for i in range(comps.shape[1])]
    pca_df = pd.DataFrame(comps, columns=comp_cols, index=df.index)

    passthrough_kept = [c for c in passthrough_cols if c in df.columns]
    out = pd.concat([df[passthrough_kept].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    PCA_OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(PCA_OUT_DB) as conn:
        # Drop intermediate transformed tables, keep only final PCA table.
        out.to_sql(PCA_OUT_TABLE, conn, if_exists="replace", index=False)
        for t in ("transformed_inf_train", "transformed_inf_validation", "transformed_inf_test"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
    print(f"[OK] Wrote PCA-transformed regression vector with passthrough cols to {PCA_OUT_DB} table {PCA_OUT_TABLE}")


def main() -> None:
    _run_log_zscore()
    _apply_saved_pca()
    print("[DONE] Regression inference vector ready.")


if __name__ == "__main__":
    main()
