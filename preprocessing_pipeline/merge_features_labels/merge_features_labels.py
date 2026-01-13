from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

LABEL_PHASE = "main_phase"

FEATURES_DB = Path(
    "/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/check_multicolinearity/all_preprocessed_sources.db"
)
LABELS_DB = Path(
    f"/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/labels/{LABEL_PHASE}_label/{LABEL_PHASE}_labels.db"
)
OUTPUT_DB = Path(
    f"/home/haris/Documents/ProjectBzarre/preprocessing_pipeline/merge_features_labels/features_with_labels.db"
)

FEATURE_TABLES = {
    "train": "merged_train",
    "validation": "merged_validation",
    "test": "merged_test",
}
LABEL_TABLES = {
    "train": f"storm_{LABEL_PHASE}_train",
    "validation": f"storm_{LABEL_PHASE}_validation",
    "test": f"storm_{LABEL_PHASE}_test",
}
LABEL_PREFIX = f"{LABEL_PHASE}_labels_"


def _load_table(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_localize(None)


def main() -> None:
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(OUTPUT_DB) as out_conn:
        for split in ("train", "validation", "test"):
            feature_table = FEATURE_TABLES[split]
            label_table = LABEL_TABLES[split]

            features = _load_table(FEATURES_DB, feature_table)
            labels = _load_table(LABELS_DB, label_table)

            if "timestamp" not in features.columns:
                raise RuntimeError(f"Missing timestamp in {FEATURES_DB}:{feature_table}")
            if "timestamp" not in labels.columns:
                raise RuntimeError(f"Missing timestamp in {LABELS_DB}:{label_table}")

            features = features.copy()
            labels = labels.copy()
            features["timestamp"] = _normalize_timestamp(features["timestamp"])
            labels["timestamp"] = _normalize_timestamp(labels["timestamp"])

            label_cols = [c for c in labels.columns if c != "timestamp"]
            labels = labels.rename(columns={c: f"{LABEL_PREFIX}{c}" for c in label_cols})

            merged = features.merge(labels, on="timestamp", how="inner")
            out_table = f"merged_{split}"
            merged.to_sql(out_table, out_conn, if_exists="replace", index=False)
            count = out_conn.execute(f"SELECT COUNT(*) FROM {out_table}").fetchone()[0]
            print(f"[OK] Created {out_table} ({count:,} rows)")

    print(f"[OK] Merged DB saved to {OUTPUT_DB}")


if __name__ == "__main__":
    main()
