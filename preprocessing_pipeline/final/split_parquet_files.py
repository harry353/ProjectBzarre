from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "space_weather_api.py").exists():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_DB = PROJECT_ROOT / "preprocessing_pipeline" / "final" / "all_sources_intersection.db"
OUTPUT_DIR = PROJECT_ROOT / "modeling_pipeline" / "data"

SPLIT_CONFIG = {
    "train": OUTPUT_DIR / "train" / "train_h6.parquet",
    "validation": OUTPUT_DIR / "validation" / "validation_h6.parquet",
    "test": OUTPUT_DIR / "test" / "test_h6.parquet",
}


def _export_split(split: str, target_path: Path) -> None:
    table = f"combined_{split}"
    with sqlite3.connect(FINAL_DB) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

    if df.empty:
        raise RuntimeError(f"Split '{split}' is empty in {FINAL_DB}")

    df = df.drop(columns=[col for col in ("timestamp", "time_tag") if col in df.columns])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_path, index=False)
    print(f"[OK] Wrote {split} split to {target_path}")


def main() -> None:
    if not FINAL_DB.exists():
        raise FileNotFoundError(f"Combined database not found at {FINAL_DB}")

    for split, path in SPLIT_CONFIG.items():
        _export_split(split, path)

    print("[OK] Parquet exports complete.")


if __name__ == "__main__":
    main()
