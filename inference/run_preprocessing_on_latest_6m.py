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

SNAPSHOT_DB = PROJECT_ROOT / "inference" / "space_weather_last_6m.db"
UPDATE_SCRIPT = PROJECT_ROOT / "inference" / "update_space_weather_last_6m.py"
PIPELINE_RUNNER = PROJECT_ROOT / "preprocessing_pipeline" / "run_full_preprocessing_pipeline.py"
MERGED_OUTPUT_DB = (
    PROJECT_ROOT
    / "preprocessing_pipeline"
    / "merge_features"
    / "all_preprocessed_sources.db"
)
OUTPUT_DIR = PROJECT_ROOT / "inference"
OUTPUT_FILENAME = "preprocessed_vector_1m.db"
KEEP_LATEST_ENTRIES = 720
SOURCE_TABLE_PREFERENCE = ["merged_test", "merged_validation", "merged_train", "original_vector"]


# Main entry point to run the preprocessing pipeline against a 6-month data snapshot
def main() -> None:
    if not UPDATE_SCRIPT.exists():
        raise FileNotFoundError(f"Update script not found: {UPDATE_SCRIPT}")
    # Ensure the snapshot database exists; trigger update if missing
    if not SNAPSHOT_DB.exists():
        print(stamp("Snapshot DB missing; rebuilding from live sources..."))
        subprocess.run([sys.executable, str(UPDATE_SCRIPT)], check=True)

    if not PIPELINE_RUNNER.exists():
        raise FileNotFoundError(f"Pipeline runner not found: {PIPELINE_RUNNER}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(stamp(f"Running preprocessing pipeline against {SNAPSHOT_DB}"))
    # Configure environment variables to direct the pipeline to use the local inference DB
    env = os.environ.copy()
    env["SPACE_WEATHER_DB_PATH"] = str(SNAPSHOT_DB)
    env["PREPROC_SKIP_SPLITS"] = "1"

    # Launch the core preprocessing pipeline script
    subprocess.run([sys.executable, str(PIPELINE_RUNNER)], check=True, env=env)

    if not MERGED_OUTPUT_DB.exists():
        raise FileNotFoundError(f"Merged output missing: {MERGED_OUTPUT_DB}")

    # Copy the merged features to the local inference directory
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    shutil.copy2(MERGED_OUTPUT_DB, output_path)
    # Simplify the database structure for inference consumption
    _collapse_to_single_table(output_path, KEEP_LATEST_ENTRIES)

    elapsed = time.time() - started
    print(stamp(f"Preprocessed snapshot written to {output_path}"))
    print(stamp(f"Pipeline run completed in {elapsed:.1f} seconds."))


# Simplify database by keeping only the most preferred table and trimming historical depth
def _collapse_to_single_table(db_path: Path, limit: int) -> None:
    # Local imports to minimize startup time of the main script
    import sqlite3
    import pandas as pd

    # Recognized column names for time-based sorting
    time_columns = ("timestamp", "time_tag", "date")
    with sqlite3.connect(db_path) as conn:
        # Retrieve names of all physical tables in the database
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        # Select the best available table based on the SOURCE_TABLE_PREFERENCE list
        source_table = None
        for candidate in SOURCE_TABLE_PREFERENCE:
            if candidate in tables:
                source_table = candidate
                break
        if source_table is None:
            if not tables:
                raise RuntimeError("No tables found to collapse.")
            source_table = sorted(tables)[0]

        # Load the chosen table and standardize the time column for sorting
        df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)
        time_col = next((c for c in time_columns if c in df.columns), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
            df = df.dropna(subset=[time_col])
            df = df.sort_values(time_col)
        
        # Apply the row limit (e.g., last 720 hours for a 1-month lookback)
        if limit and limit > 0 and not df.empty:
            df = df.iloc[-limit:]

        # Clear out all existing tables to make way for the unified inference table
        existing_tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for tbl in existing_tables:
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        
        # Write the final processed vector to the consolidated table name
        df.to_sql("inference_vector", conn, if_exists="replace", index=False)
        conn.commit()

if __name__ == "__main__":
    main()
