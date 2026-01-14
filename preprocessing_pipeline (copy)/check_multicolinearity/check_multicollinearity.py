from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


DESIGN_MATRIX_PATH = Path(
    "/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/check_multicolinearity/all_preprocessed_sources.db"
)
DESIGN_MATRIX_TABLE = "merged_train"
OUTPUT_DIR = Path(__file__).resolve().parent / "multicollinearity_diagnostics"

CORR_THRESHOLD = 0.8
VIF_WARN_THRESHOLD = 5.0
VIF_HIGH_THRESHOLD = 10.0
COND_THRESHOLD = 30.0
SINGULAR_REL_TOL = 1e-6
SINGULAR_ABS_TOL = 1e-12


def _load_design_matrix(path: Path, table: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Design matrix not found at {path}")
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    df = _load_design_matrix(DESIGN_MATRIX_PATH, DESIGN_MATRIX_TABLE)
    df = df[[c for c in df.columns if "labels" not in c]]
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.dropna(axis=0, how="any")
    if numeric_df.empty:
        raise RuntimeError("No numeric rows remain after dropping missing values.")

    scaler = StandardScaler()
    X_values = scaler.fit_transform(numeric_df.to_numpy(dtype=float))
    X = pd.DataFrame(X_values, columns=numeric_df.columns)

    corr = X.corr()
    corr_path = OUTPUT_DIR / "correlation_matrix.csv"

    corr_pairs = []
    cols = corr.columns.tolist()
    corr_values = corr.to_numpy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = float(corr_values[i, j])
            abs_value = abs(value)
            if abs_value >= CORR_THRESHOLD:
                corr_pairs.append((cols[i], cols[j], value, abs_value))
    corr_pairs.sort(key=lambda x: (-x[3], x[0], x[1]))
    corr_pairs_df = pd.DataFrame(
        corr_pairs,
        columns=["feature_1", "feature_2", "correlation", "abs_correlation"],
    )
    corr_pairs_path = OUTPUT_DIR / "high_correlation_pairs.csv"

    exog = add_constant(X.to_numpy(), has_constant="add")
    vif_values = []
    for idx, name in enumerate(X.columns, start=1):
        vif = float(variance_inflation_factor(exog, idx))
        vif_values.append((name, vif))
    vif_df = pd.DataFrame(vif_values, columns=["feature", "vif"])
    vif_df = vif_df.sort_values(by="vif", ascending=False).reset_index(drop=True)
    vif_path = OUTPUT_DIR / "vif.csv"

    singular_values = np.linalg.svd(X.to_numpy(), compute_uv=False)
    singular_df = pd.DataFrame(
        {"index": np.arange(len(singular_values)), "singular_value": singular_values}
    )
    singular_path = OUTPUT_DIR / "singular_values.csv"

    max_sv = float(np.max(singular_values))
    min_sv = float(np.min(singular_values))
    condition_number = float("inf") if min_sv == 0.0 else max_sv / min_sv
    singular_threshold = max(max_sv * SINGULAR_REL_TOL, SINGULAR_ABS_TOL)
    near_degenerate = singular_values <= singular_threshold
    near_degenerate_indices = np.where(near_degenerate)[0].tolist()
    near_degenerate_values = singular_values[near_degenerate].tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(corr_path, index=True)
    corr_pairs_df.to_csv(corr_pairs_path, index=False)
    vif_df.to_csv(vif_path, index=False)
    singular_df.to_csv(singular_path, index=False)

    vif_warn = vif_df[vif_df["vif"] >= VIF_WARN_THRESHOLD]
    vif_high = vif_df[vif_df["vif"] >= VIF_HIGH_THRESHOLD]

    summary = {
        "design_matrix_path": str(DESIGN_MATRIX_PATH),
        "row_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "design_matrix_table": DESIGN_MATRIX_TABLE,
        "correlation_threshold": CORR_THRESHOLD,
        "high_correlation_pair_count": int(len(corr_pairs_df)),
        "high_correlation_pairs": corr_pairs_df.to_dict(orient="records"),
        "vif_thresholds": {
            "warn": VIF_WARN_THRESHOLD,
            "high": VIF_HIGH_THRESHOLD,
        },
        "vif_warn_count": int(len(vif_warn)),
        "vif_high_count": int(len(vif_high)),
        "vif_warn_features": vif_warn["feature"].tolist(),
        "vif_high_features": vif_high["feature"].tolist(),
        "condition_number": condition_number,
        "condition_number_threshold": COND_THRESHOLD,
        "singular_value_threshold": singular_threshold,
        "near_degenerate_count": int(len(near_degenerate_values)),
        "near_degenerate_indices": near_degenerate_indices,
        "near_degenerate_values": near_degenerate_values,
    }

    summary_json_path = OUTPUT_DIR / "summary.json"
    summary_txt_path = OUTPUT_DIR / "summary.txt"
    with summary_json_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    multicollinearity_present = (
        len(corr_pairs_df) > 0
        or len(vif_high) > 0
        or condition_number >= COND_THRESHOLD
        or len(near_degenerate_values) > 0
    )

    top_vif = vif_df.head(5).to_dict(orient="records")
    top_pairs = corr_pairs_df.head(5).to_dict(orient="records")

    top_vif_text = ", ".join(
        "{}({:.2f})".format(row["feature"], row["vif"]) for row in top_vif
    ) or "None"
    top_pairs_text = ", ".join(
        "{}~{}({:.2f})".format(row["feature_1"], row["feature_2"], row["correlation"])
        for row in top_pairs
    ) or "None"

    summary_lines = [
        f"Multicollinearity: {'PRESENT' if multicollinearity_present else 'NOT DETECTED'}",
        f"Rows: {len(X)}, Features: {X.shape[1]}",
        f"Condition number: {condition_number:.4g}",
        f"High-correlation pairs (|r| >= {CORR_THRESHOLD}): {len(corr_pairs_df)}/{int(X.shape[1] * (X.shape[1] - 1) / 2)}",
        f"VIF >= {VIF_WARN_THRESHOLD}: {len(vif_warn)}; VIF >= {VIF_HIGH_THRESHOLD}: {len(vif_high)}",
        f"Near-degenerate singular values (<= {singular_threshold:.4g}): {len(near_degenerate_values)}",
        f"Top VIF features: {top_vif_text}",
        f"Top correlation pairs: {top_pairs_text}",
    ]

    summary_txt_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
