from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_PIPELINE = PROJECT_ROOT / "ml_pipeline"

SCRIPTS = [
    ML_PIPELINE / "tune_hazard_xgboost.py",
    ML_PIPELINE / "export_raw_probabilities.py",
    ML_PIPELINE / "probability_calibration.py",
    # ML_PIPELINE / "build_storm_onset_distribution.py",
]


def main() -> None:
    python = sys.executable
    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        print(f"[RUN] {script}")
        subprocess.run([python, str(script)], check=True)


if __name__ == "__main__":
    main()
