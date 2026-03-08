from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_PIPELINE = PROJECT_ROOT / "classification_pipeline_"

# Ordered list of scripts to execute for a full model run
SCRIPTS = [
    ML_PIPELINE / "train_m odel.py",
    ML_PIPELINE / "export_raw_probabilities.py",
    ML_PIPELINE / "probability_calibration.py",
]


# Iterate through the configured scripts and execute them sequentially
def main() -> None:
    python = sys.executable
    # Orchestration loop: running training, export, and calibration in order
    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")
        print(f"[RUN] {script}")
        # Execute script and block until completion; raise error if script fails
        subprocess.run([python, str(script)], check=True)


if __name__ == "__main__":
    main()
