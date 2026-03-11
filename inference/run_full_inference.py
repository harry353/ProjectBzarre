from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DB = True
CLEAN_DIRS = [
    PROJECT_ROOT / "preprocessing_pipeline" / "cme",
    PROJECT_ROOT / "preprocessing_pipeline" / "dst",
    PROJECT_ROOT / "preprocessing_pipeline" / "imf_solar_wind",
    PROJECT_ROOT / "preprocessing_pipeline" / "kp_index",
    PROJECT_ROOT / "preprocessing_pipeline" / "radio_flux",
    PROJECT_ROOT / "preprocessing_pipeline" / "sunspot_number",
    PROJECT_ROOT / "preprocessing_pipeline" / "xray_flux",
]

SCRIPTS = [
    PROJECT_ROOT / "inference" / "backup_swpc_imf.py",
    PROJECT_ROOT / "inference" / "update_space_weather_last_6m.py",
    PROJECT_ROOT / "inference" / "insert_swpc_imf_backup.py",
    PROJECT_ROOT / "inference" / "run_preprocessing_on_latest_6m.py",
    PROJECT_ROOT / "inference" / "classification" / "create_classification_vector.py",
    PROJECT_ROOT / "inference" / "classification" / "run_classification_inference.py",
    PROJECT_ROOT / "inference" / "classification" / "plot_storm_probability.py",
    PROJECT_ROOT / "inference" / "regression" / "create_regression_vector.py",
    PROJECT_ROOT / "inference" / "regression" / "run_regression_inference.py",
    PROJECT_ROOT / "inference" / "regression" / "plot_predicted_dst.py",
    PROJECT_ROOT / "inference" / "combined_plot" / "plot_combined.py",
]


# Helper to execute a script as a separate process and wait for completion
def _run(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    print(f"[RUN] {script}")
    # Execute script using the current Python interpreter
    subprocess.run([sys.executable, str(script)], check=True)


# Scrub intermediate database files from preprocessing directories
def _clean_db_artifacts() -> None:
    removed = 0
    for folder in CLEAN_DIRS:
        if not folder.exists():
            continue
        # Recursively find and delete all .db files
        for db_file in folder.rglob("*.db"):
            try:
                db_file.unlink()
                removed += 1
            except Exception as exc:
                print(f"[WARN] Could not remove {db_file}: {exc}")
    if removed:
        print(f"[CLEANUP] Removed {removed} .db files from preprocessing directories.")


# Orchestrate the end-to-end inference pipeline: Data Update -> Preprocessing -> Classification -> Regression
def main() -> None:
    start = time.time()
    # Execute the sequence of scripts defined in SCRIPTS
    for script in SCRIPTS:
        _run(script)

    # Manual cleanup of specific intermediate vector databases
    for fname in (
        # "classification_horizons_vector_1m.db",
        # "preprocessed_vector_1m.db",
        # "classification/classification_horizons_vector_1m.db",
        # "regression/pca_regression_vector_1m.db",
    ):
        path = PROJECT_ROOT / "inference" / fname
        if path.exists():
            try:
                path.unlink()
                print(f"[CLEANUP] Removed {path}")
            except Exception as exc:
                print(f"[WARN] Could not remove {path}: {exc}")

    # Broad cleanup of preprocessing artifacts if configured
    if CLEAN_DB:
        _clean_db_artifacts()

    elapsed = time.time() - start
    print(f"[OK] Inference pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
