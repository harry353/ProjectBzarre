from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DB = PROJECT_ROOT / "inference" / "inference_vector.db"
OUTPUT_DB = PROJECT_ROOT / "inference" / "horizon_vector.db"
HORIZON_MODELS_DIR = PROJECT_ROOT / "ml_pipeline" / "horizon_models"

# Table preference order if multiple merged tables exist
SOURCE_TABLE_PREFERENCE = [
    "inference_vector",
    "merged_test",
    "merged_validation",
    "merged_train",
]


def _load_features(horizon: int) -> list[str]:
    path = HORIZON_MODELS_DIR / f"h{horizon}" / "selected_features.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected_features.json for h{horizon}: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"selected_features.json for h{horizon} must contain a list.")
    return [str(item) for item in data]


def _choose_source_table(conn: sqlite3.Connection) -> str:
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM inputdb.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    for name in SOURCE_TABLE_PREFERENCE:
        if name in tables:
            return name
    if not tables:
        raise RuntimeError("No tables found in input DB.")
    return sorted(tables)[0]


def _available_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA inputdb.table_info({table})")]


def _build_table(
    conn_out: sqlite3.Connection,
    source_table: str,
    dest_table: str,
    keep_cols: list[str],
) -> None:
    columns_sql = ", ".join(keep_cols)
    conn_out.execute(f"DROP TABLE IF EXISTS {dest_table}")
    conn_out.execute(
        f"CREATE TABLE {dest_table} AS SELECT {columns_sql} FROM inputdb.{source_table}"
    )


def main() -> None:
    if not INPUT_DB.exists():
        raise FileNotFoundError(f"Input DB not found: {INPUT_DB}")

    OUTPUT_DB.unlink(missing_ok=True)
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(OUTPUT_DB) as conn_out:
        conn_out.execute(f"ATTACH DATABASE '{INPUT_DB}' AS inputdb")

        source_table = _choose_source_table(conn_out)
        available = _available_columns(conn_out, source_table)

        # Copy full source table as original_vector
        conn_out.execute("DROP TABLE IF EXISTS original_vector")
        conn_out.execute(
            f"CREATE TABLE original_vector AS SELECT * FROM inputdb.{source_table}"
        )

        timestamp_cols = ["timestamp", "time_tag", "date"]
        ts_col = next((c for c in timestamp_cols if c in available), None)

        for horizon in range(1, 9):
            features = _load_features(horizon)
            keep = []
            if ts_col:
                keep.append(ts_col)
            keep.extend([f for f in features if f in available and f not in keep])

            if not keep:
                print(f"[WARN] h{horizon}: no matching columns; table skipped.")
                continue

            _build_table(conn_out, source_table, f"h{horizon}_vector", keep)
            print(
                f"[OK] h{horizon}: wrote table with columns {keep} from source '{source_table}'."
            )
        conn_out.commit()


if __name__ == "__main__":
    main()
