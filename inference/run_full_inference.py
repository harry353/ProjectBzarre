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
    PROJECT_ROOT / "inference" / "update_space_weather_last_6m.py",
    PROJECT_ROOT / "inference" / "run_preprocessing_on_latest_6m.py",
    PROJECT_ROOT / "inference" / "create_horizon_vector.py",
    PROJECT_ROOT / "inference" / "run_horizon_models.py",
    PROJECT_ROOT / "inference" / "plot_predicted_probabilities.py",
]


def _run(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    print(f"[RUN] {script}")
    subprocess.run([sys.executable, str(script)], check=True)


def _clean_db_artifacts() -> None:
    removed = 0
    for folder in CLEAN_DIRS:
        if not folder.exists():
            continue
        for db_file in folder.rglob("*.db"):
            try:
                db_file.unlink()
                removed += 1
            except Exception as exc:
                print(f"[WARN] Could not remove {db_file}: {exc}")
    if removed:
        print(f"[CLEANUP] Removed {removed} .db files from preprocessing directories.")


def main() -> None:
    start = time.time()
    for script in SCRIPTS:
        _run(script)
    # Clean up intermediate DBs
    for fname in ("horizon_vector.db", "inference_vector.db"):
        path = PROJECT_ROOT / "inference" / fname
        if path.exists():
            try:
                path.unlink()
                print(f"[CLEANUP] Removed {path}")
            except Exception as exc:
                print(f"[WARN] Could not remove {path}: {exc}")
    if CLEAN_DB:
        _clean_db_artifacts()
    elapsed = time.time() - start
    print(f"[OK] Inference pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
