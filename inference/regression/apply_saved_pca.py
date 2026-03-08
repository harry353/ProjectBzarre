from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_DB = PROJECT_ROOT / "inference" / "preprocessed_vector_1m.db"
IN_TABLE = "inference_vector"
OUT_DB = PROJECT_ROOT / "inference" / "regression" / "pca_features_1m.db"
OUT_TABLE = "pca_inference_vector"
PCA_MODEL_PATH = PROJECT_ROOT / "regression_pipeline" / "pca_model.joblib"
PCA_COLUMNS_PATH = PROJECT_ROOT / "regression_pipeline" / "pca_columns.json"
PASSTHROUGH_COLS = ["timestamp", "time_tag", "date"]


# Helper to load a specific database table into a pandas DataFrame
def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


# Main logic to load a pre-trained PCA model and transform the live inference vector
def main() -> None:
    # 1. Verification of required input files and model artifacts
    if not IN_DB.exists():
        raise FileNotFoundError(f"Input DB not found: {IN_DB}")
    if not PCA_MODEL_PATH.exists() or not PCA_COLUMNS_PATH.exists():
        raise FileNotFoundError("PCA model/columns not found. Run apply_pca_transforms.py first.")

    # 2. Load the PCA model (joblib) and the exact column sequence (JSON) used during training
    pca = joblib.load(PCA_MODEL_PATH)
    pca_cols = json.loads(PCA_COLUMNS_PATH.read_text(encoding="utf-8"))

    # 3. Retrieve raw inference data
    df = _load_table(IN_DB, IN_TABLE)
    if df.empty:
        raise RuntimeError(f"Input table '{IN_TABLE}' is empty.")
    
    # Ensure all features the PCA model expects are present in the current database
    missing = [c for c in pca_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required PCA columns: {missing}")

    # 4. Perform PCA transformation
    # Extract the subset of features and convert to numpy for the sklearn transformer
    X = df[pca_cols].to_numpy(dtype=float)
    comps = pca.transform(X)
    
    # Generate new column names (pc_1, pc_2, etc.) for the principal components
    comp_cols = [f"pc_{i + 1}" for i in range(comps.shape[1])]
    pca_df = pd.DataFrame(comps, columns=comp_cols, index=df.index)

    # 5. Merge components with identifying metadata (timestamps)
    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    out = pd.concat([df[passthrough].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    # 6. Persist the lower-dimensional representation to the regression output database
    OUT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(OUT_DB) as conn:
        out.to_sql(OUT_TABLE, conn, if_exists="replace", index=False)
    
    print(f"[OK] Wrote PCA-transformed inference vector to {OUT_DB} table {OUT_TABLE}")


if __name__ == "__main__":
    main()
