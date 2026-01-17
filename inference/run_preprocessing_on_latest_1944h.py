from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database_builder.logging_utils import stamp

SNAPSHOT_DB = PROJECT_ROOT / "inference" / "space_weather_last_1944h.db"
UPDATE_SCRIPT = PROJECT_ROOT / "inference" / "update_space_weather_last_1944h.py"
PIPELINE_RUNNER = PROJECT_ROOT / "preprocessing_pipeline" / "run_full_preprocessing_pipeline.py"
MERGED_OUTPUT_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "check_multicolinearity"
    / "all_preprocessed_sources.db"
)
OUTPUT_DIR = PROJECT_ROOT / "inference"
OUTPUT_FILENAME = "inference_vector.db"


def main() -> None:
    if not UPDATE_SCRIPT.exists():
        raise FileNotFoundError(f"Update script not found: {UPDATE_SCRIPT}")
    if not SNAPSHOT_DB.exists():
        print(stamp("Snapshot DB missing; rebuilding from live sources..."))
        subprocess.run([sys.executable, str(UPDATE_SCRIPT)], check=True)

    if not PIPELINE_RUNNER.exists():
        raise FileNotFoundError(f"Pipeline runner not found: {PIPELINE_RUNNER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(stamp(f"Running preprocessing pipeline against {SNAPSHOT_DB}"))
    env = os.environ.copy()
    env["SPACE_WEATHER_DB_PATH"] = str(SNAPSHOT_DB)
    env["PREPROC_SKIP_SPLITS"] = "1"

    subprocess.run([sys.executable, str(PIPELINE_RUNNER)], check=True, env=env)

    if not MERGED_OUTPUT_DB.exists():
        raise FileNotFoundError(f"Merged output missing: {MERGED_OUTPUT_DB}")

    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    shutil.copy2(MERGED_OUTPUT_DB, output_path)
    _keep_last_entry(output_path)

    elapsed = time.time() - started
    print(stamp(f"Preprocessed snapshot written to {output_path}"))
    print(stamp(f"Pipeline run completed in {elapsed:.1f} seconds."))


def _keep_last_entry(db_path: Path) -> None:
    """Trim each table to its most recent row by timestamp-like column."""
    import sqlite3

    time_columns = ("timestamp", "time_tag", "date")
    with sqlite3.connect(db_path) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            time_col = next((c for c in time_columns if c in cols), None)
            if not time_col:
                continue
            row = conn.execute(
                f"SELECT {time_col} FROM {table} ORDER BY datetime({time_col}) DESC LIMIT 1"
            ).fetchone()
            if not row or row[0] is None:
                continue
            latest = row[0]
            conn.execute(
                f"DELETE FROM {table} WHERE datetime({time_col}) < datetime(?) OR {time_col} IS NULL",
                (latest,),
            )
        conn.commit()


if __name__ == "__main__":
    main()
